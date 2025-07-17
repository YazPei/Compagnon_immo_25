import streamlit as st
import requests
import logging
from typing import Dict, Any, Optional

API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIClient:
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = TIMEOUT
    
    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def estimate_price(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/estimation", 
                json=data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            st.error(f"Erreur de communication avec l'API : {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            st.error(f"Erreur inattendue : {e}")
            return None


def validate_inputs(surface: float, pieces: int, **kwargs) -> bool:
    if surface <= 0:
        st.error("La surface doit être positive")
        return False
    if pieces <= 0:
        st.error("Le nombre de pièces doit être positif")
        return False
    if surface > 1000:
        st.warning("Surface inhabituellement grande (>1000m²)")
    return True


def format_price(price: float) -> str:
    return f"{price:,.0f}".replace(",", " ") + " €"


def display_estimation_results(result: Dict[str, Any]) -> None:
    try:
        st.success(f"**Prix estimé : {format_price(result['prix_estime'])}**")
        
        if 'fourchette' in result:
            fourchette = result['fourchette']
            st.info(
                f"Fourchette : {format_price(fourchette['min'])} - "
                f"{format_price(fourchette['max'])}"
            )
        
        if 'marche' in result:
            marche = result['marche']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Prix moyen du secteur", 
                    format_price(marche['prix_moyen_secteur'])
                )
            with col2:
                delai = marche.get('delai_vente_moyen', 'N/A')
                st.metric("Délai de vente moyen", f"{delai} jours")
            
            evol = marche.get('evolution_annuelle', 0)
            if abs(evol) < 0.1:
                st.write("📊 **Évolution annuelle** : Stable")
            elif evol > 0:
                st.write(f"📈 **Évolution annuelle** : +{evol:.1f}% (hausse)")
            else:
                st.write(f"📉 **Évolution annuelle** : {evol:.1f}% (baisse)")
        
        with st.expander("ℹ️ Détails de l'estimation"):
            metadata = result.get('metadata', {})
            st.write(f"**Date d'estimation :** {metadata.get('date_estimation', 'N/A')}")
            st.write(f"**Version du modèle :** {metadata.get('version_modele', 'N/A')}")
            st.write(f"**ID estimation :** {metadata.get('id_estimation', 'N/A')}")
            
            if 'explications' in result:
                st.markdown("#### Explications :")
                for key, value in result['explications'].items():
                    st.markdown(f"**{key}** : {value}")
    
    except KeyError as e:
        st.error(f"Données manquantes dans la réponse : {e}")
    except Exception as e:
        st.error(f"Erreur lors de l'affichage : {e}")


def main():
    """Interface principale Streamlit."""
    st.set_page_config(
        page_title="Compagnon d'Estimation Immobilière",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 Compagnon d'Estimation Immobilière")
    st.markdown("---")
    
    api_client = APIClient()
    
    if not api_client.health_check():
        st.error("⚠️ L'API n'est pas accessible. Vérifiez qu'elle est démarrée.")
        st.stop()
    else:
        st.success("✅ API connectée")
    
    with st.form("estimation_form"):
        st.subheader("📝 Informations sur le bien")
        
        col1, col2 = st.columns(2)
        with col1:
            surface = st.number_input(
                "Surface habitable (m²)", 
                min_value=1.0, 
                max_value=1000.0, 
                value=70.0,
                step=1.0
            )
            pieces = st.number_input(
                "Nombre de pièces principales", 
                min_value=1, 
                max_value=20, 
                value=3
            )
        
        with col2:
            classe_energie = st.selectbox(
                "Classe énergétique",
                ["A", "B", "C", "D", "E", "F", "G", "Non renseigné"]
            )
        
        submitted = st.form_submit_button("Estimer le prix")
        
        if submitted:
            if validate_inputs(surface, pieces):
                with st.spinner("Estimation en cours..."):
                    data = {
                        "surface_reelle_bati": surface,
                        "nombre_pieces_principales": pieces,
                        "classe_energetique": (
                            classe_energie if classe_energie != "Non renseigné" else None
                        )
                    }
                    
                    result = api_client.estimate_price(data)
                    if result:
                        display_estimation_results(result)


if __name__ == "__main__":
    main()