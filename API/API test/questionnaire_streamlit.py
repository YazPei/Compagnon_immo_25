import streamlit as st
import requests

st.set_page_config(page_title="Estimation Immobilière", page_icon="🏠", layout="centered")
st.title("Estimation Immobilière en ligne 🏠")
st.markdown("Remplissez les champs ci-dessous : le formulaire s'adapte au type de bien pour rester léger.")

API_URL = "http://127.0.0.1:8001/api/v1/questionnaire_estimation"
API_KEY = "test-key-123"

# Sélection du type de bien (hors du formulaire pour déclencher le refresh dynamique)
type_bien = st.selectbox("Type de bien", ["appartement", "maison", "autre"], key="type_bien")

# --- Bloc 2 : Localisation dynamique (en dehors du formulaire) ---
st.subheader("Localisation")
adresse_input = st.text_input("Commencez à taper l'adresse...", key="adresse_input")
suggestions = []
suggestions_data = []
if adresse_input and len(adresse_input) > 3:
    url = f"https://api-adresse.data.gouv.fr/search/?q={adresse_input}&limit=5"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            suggestions = [f['properties']['label'] for f in data['features']]
            suggestions_data = data['features']
    except Exception as e:
        st.warning(f"Erreur lors de la recherche d'adresse : {e}")
selected_idx = 0
if suggestions:
    adresse = st.selectbox("Suggestions d'adresses :", suggestions, key="adresse_suggestion")
    selected_idx = suggestions.index(adresse)
    # On stocke code postal et ville dans session_state
    if suggestions_data:
        props = suggestions_data[selected_idx]['properties']
        st.session_state['code_postal_auto'] = props.get('postcode', '')
        st.session_state['ville_auto'] = props.get('city', '')
        latitude, longitude = props.get('y', None), props.get('x', None)
else:
    adresse = adresse_input
    st.session_state['code_postal_auto'] = ''
    st.session_state['ville_auto'] = ''

with st.form("questionnaire"):
    # --- Bloc 1 : Informations de base ------------------------------------
    st.subheader("Informations de base")
    col1, col2 = st.columns(2)
    with col1:
        surface = st.number_input("Surface habitable (m²)", 1, 1000, 55)
        nb_pieces = st.number_input("Nombre de pièces", 1, 20, 3)
        nb_sdb = st.number_input("Nombre de salles de bain", 0, 5, 1)
        bain = nb_sdb  # Pour correspondre à la feature du modèle
        eau = st.number_input("Nombre de points d'eau (douche, salle d'eau, etc.)", 0, 5, 0)
    with col2:
        nb_chambres = st.number_input("Nombre de chambres", 0, 10, 2)
        annee_construction = st.number_input("Année de construction", 1800, 2100, 1970)
        # Affichage dynamique du champ surface_terrain pour 'maison'
        surface_terrain = 0
        if type_bien == "maison":
            surface_terrain = st.number_input("Surface du terrain (m²)", 0, 10000, 0, key="surface_terrain")
        if type_bien == "appartement":
            etage = st.number_input("Étage du bien", 0, 50, 0)
        else:
            etage = 0  # placeholder

    # --- Bloc 3 : Performance & confort -----------------------------------
    st.subheader("Performance & confort")
    col3, col4 = st.columns(2)
    with col3:
        dpeL = st.selectbox("Classe énergétique (DPE)", list("ABCDEFG"))
        ascenseur = st.checkbox("Ascenseur", True)
        balcon = st.checkbox("Balcon", False)
        terrasse = st.checkbox("Terrasse", False)
    with col4:
        parking = st.checkbox("Parking", False)
        cave = st.checkbox("Cave", True)
        piscine = st.checkbox("Piscine", False)
        gardien = st.checkbox("Gardien", False)

    # --- Bloc 4 : Chauffage (expander pour ne pas alourdir) ---------------
    with st.expander("Détails chauffage"):
        energie = st.multiselect(
            "Énergie(s) utilisée(s)",
            ["Électrique", "Gaz", "Fioul", "Bois"],
            default=[]
        )
        mode = st.selectbox(
            "Mode de chauffage",
            ["Individuel", "Collectif", "Central", "Individuel, Central"]
        )

    # --- Bloc 5 : Avancé ---------------------------------------------------
    with st.expander("Options avancées"):
        typedetransaction = st.selectbox(
            "Type de transaction",
            ["vente (particulier)", "vente (promoteur)", "particulier à investisseur"]
        )
        force_cluster = st.selectbox(
            "Forcer un cluster SARIMAX",
            [None, 0, 1, 2, 3, -1],
            format_func=lambda x: "(auto)" if x is None else f"Cluster {x}"
        )

    submitted = st.form_submit_button("Obtenir mon estimation")

# ------------------ Envoi à l'API ------------------
if submitted:
    chauffage_dict = {
        "chauffage_energie_Électrique": int("Électrique" in energie),
        "chauffage_energie_Gaz": int("Gaz" in energie),
        "chauffage_energie_Fioul": int("Fioul" in energie),
        "chauffage_energie_Bois": int("Bois" in energie),
        # modes
        "chauffage_mode_Individuel":        int(mode == "Individuel"),
        "chauffage_mode_Collectif":         int(mode == "Collectif"),
        "chauffage_mode_Central":           int(mode == "Central"),
        "chauffage_mode_Individuel__Central": int(mode == "Individuel, Central")
    }

    payload = {
        # --- features saisies ---
        "type_bien": type_bien,
        "surface": surface,
        "nb_pieces": nb_pieces,
        "nb_chambres": nb_chambres,
        "bain": bain,
        "eau": eau,
        "surface_terrain": float(surface_terrain),
        "annee_construction": annee_construction,
        "etage": etage,
        "adresse": st.session_state.get("adresse_suggestion", adresse),
        "code_postal": st.session_state.get("code_postal_auto", ""),
        "ville": st.session_state.get("ville_auto", ""),
        "latitude": latitude,
        "longitude": longitude,
        "dpeL": dpeL,
        "ascenseur": ascenseur,
        "balcon": balcon,
        "terrasse": terrasse,
        "parking": parking,
        "cave": cave,
        "piscine": piscine,
        "gardien": gardien,
        "typedetransaction": typedetransaction.split()[0],  # vp / v / pi
        # --- chauffage one-hots ---
        **chauffage_dict
    }
    if force_cluster is not None:
        payload["forced_cluster"] = force_cluster

    # appel API
    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"X-API-Key": API_KEY}
        )
        if response.status_code == 200:
            result = response.json()
            st.markdown("## 🏡 Résultat de votre estimation")
            st.success(f"**Prix estimé :** {result['estimation']['prix']:.0f} €")
            st.write(f"Fourchette de prix : {result['estimation']['prix_min']:.0f} € — {result['estimation']['prix_max']:.0f} €")
            st.write(f"Prix au m² estimé : {result['estimation']['prix_m2']:.0f} €/m²")
            st.write(f"Indice de confiance : {result['estimation']['indice_confiance']} / 100")

            st.markdown("### 📊 Tendances du marché")
            # Prix moyen au m² dans le quartier
            prix_moyen = result['marche']['prix_moyen_quartier']
            if prix_moyen is None or prix_moyen < 500:
                st.write("Prix moyen au m² dans le quartier : Non disponible")
            else:
                st.write(f"Prix moyen au m² dans le quartier : {prix_moyen:,.0f} €/m²")

            # Évolution annuelle estimée
            evol = result['marche']['evolution_annuelle']
            if abs(evol) < 0.1:
                st.write("Évolution annuelle estimée : Stable")
            elif evol > 0:
                st.write(f"Évolution annuelle estimée : +{evol:.1f} % (hausse)")
            else:
                st.write(f"Évolution annuelle estimée : {evol:.1f} % (baisse)")

            st.write(f"Délai de vente moyen : {result['marche']['delai_vente_moyen']} jours")

            with st.expander("ℹ️ Détails de l'estimation et explications"):
                st.write("**Date de l'estimation :**", result['metadata']['date_estimation'])
                st.write("**Version du modèle :**", result['metadata']['version_modele'])
                st.write("**ID estimation :**", result['metadata']['id_estimation'])
                st.markdown("#### Explications :")
                for k, v in result['explications'].items():
                    st.markdown(f"**{k}** : {v}")

            # Bloc debug (optionnel, à masquer pour l'utilisateur final)
            # with st.expander("🛠️ Données techniques (debug)"):
            #     st.json(result)
        else:
            st.error(f"Erreur {response.status_code} : {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")