from typing import List, Optional

from pydantic import BaseModel, Field


class BienModel(BaseModel):
    type: str
    surface: float
    nb_pieces: int
    nb_chambres: int
    etage: int
    annee_construction: int
    etat_general: str
    exposition: str
    ascenseur: bool
    balcon: bool
    terrasse: bool
    surface_exterieure: Optional[float] = None
    parking: bool
    cave: bool
    dpe: Optional[str] = None


class LocalisationModel(BaseModel):
    code_postal: str
    ville: str
    quartier: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class TransactionModel(BaseModel):
    type: str


class EstimationRequest(BaseModel):
    bien: BienModel
    localisation: LocalisationModel
    transaction: TransactionModel


class EstimationResultModel(BaseModel):
    prix: float
    prix_min: float
    prix_max: float
    prix_m2: float
    indice_confiance: int


class MarcheModel(BaseModel):
    prix_moyen_quartier: float
    evolution_annuelle: float
    delai_vente_moyen: int


class MetadataModel(BaseModel):
    id_estimation: str
    date_estimation: str
    version_modele: str


class EstimationResponse(BaseModel):
    estimation: EstimationResultModel
    marche: MarcheModel
    metadata: MetadataModel
    explications: dict = None
    exogenes_utilisees: dict = None
    features_utilisees_prix: dict = None
    debug_info: dict = None


class ErrorResponse(BaseModel):
    error: str
    missing_fields: Optional[List[str]] = None
    invalid_fields: Optional[List[str]] = None


class TooManyRequestsResponse(BaseModel):
    error: str
    retry_after: float


class HistoriqueItemModel(BaseModel):
    id_estimation: str
    date_estimation: str
    bien: dict
    prix_estime: float
    indice_confiance: int


class HistoriqueResponse(BaseModel):
    estimations: List[HistoriqueItemModel]
    metadata: dict


class QuestionnaireRequest(BaseModel):
    type_bien: str = Field(..., description="Type de bien (appartement, maison, etc.)")
    surface: float = Field(..., description="Surface habitable en m²")
    nb_pieces: int = Field(..., description="Nombre de pièces principales")
    nb_chambres: int = Field(..., description="Nombre de chambres")
    adresse: str = Field(
        ..., description="Adresse complète du bien (numéro, rue, etc.)"
    )
    code_postal: str = Field(..., description="Code postal du bien")
    ville: str = Field(..., description="Ville du bien")
    annee_construction: Optional[int] = Field(
        None, description="Année de construction (optionnel)"
    )
    ascenseur: bool = Field(False, description="Présence d'un ascenseur")
    balcon: bool = Field(False, description="Présence d'un balcon")
    parking: bool = Field(False, description="Présence d'un parking")
    cave: bool = Field(False, description="Présence d'une cave")
    gardien: bool = Field(False, description="Présence d'un gardien")
    piscine: bool = Field(False, description="Présence d'une piscine")
    terrasse: bool = Field(False, description="Présence d'une terrasse")
    surface_terrain: Optional[float] = Field(
        0, description="Surface du terrain (optionnel, maisons)"
    )
    dpeL: Optional[str] = Field(None, description="Classe énergétique (DPE) : A à G")
    bain: Optional[int] = Field(0, description="Nombre de salles de bain")
    eau: Optional[int] = Field(
        0, description="Nombre de points d'eau (salle d'eau, douche, etc.)"
    )
    etage: Optional[int] = Field(0, description="Étage du bien (optionnel)")
    # Champs avancés pour le cluster ou le type de transaction
    typedetransaction: Optional[str] = Field(
        None, description="Type de transaction (vp, v, pi, etc.)"
    )
    forced_cluster: Optional[int] = Field(
        None, description="Forcer un cluster SARIMAX (optionnel)"
    )
