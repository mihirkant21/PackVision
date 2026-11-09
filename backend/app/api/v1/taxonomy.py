from fastapi import APIRouter
from backend.app.schemas.taxonomy import (
    TaxonomyResponse,
    TaxonomyItem,
    MaterialItem,
)

router = APIRouter()

TAXONOMY_DATA = TaxonomyResponse(
    package_types=[
        TaxonomyItem(
            id="bottle",
            label="Bottle",
            description="Glass or plastic bottles for beverages, spirits, oils, and skincare serums.",
            icon="bottle",
        ),
        TaxonomyItem(
            id="box",
            label="Box / Carton",
            description="Rigid, folding, and corrugated gift boxes, retail cartons, and luxury packaging.",
            icon="package",
        ),
        TaxonomyItem(
            id="can",
            label="Can",
            description="Aluminum beverage cans, aerosol cans, and tin canisters.",
            icon="disc",
        ),
        TaxonomyItem(
            id="pouch",
            label="Stand-Up Pouch",
            description="Flexible doy-pack stand-up pouches with zip seals for food, powders, and coffee.",
            icon="shopping-bag",
        ),
        TaxonomyItem(
            id="jar",
            label="Jar",
            description="Cosmetic cream jars and culinary glass preserve containers.",
            icon="circle",
        ),
        TaxonomyItem(
            id="tube",
            label="Squeeze Tube",
            description="Flexible squeeze tubes for lotions, creams, toothpastes, and cosmetics.",
            icon="cylinder",
        ),
    ],
    materials=[
        MaterialItem(
            id="cardboard",
            label="Cardboard / Paperboard",
            default_finish="matte",
            compatible_finishes=["matte", "glossy", "embossed", "foil_stamping"],
            description="Sustainable folding boxboard with sharp print reproduction.",
        ),
        MaterialItem(
            id="glass",
            label="Glass",
            default_finish="glossy",
            compatible_finishes=["glossy", "matte", "frosted", "embossed"],
            description="Premium clear or tinted glass with realistic refractions and caustics.",
        ),
        MaterialItem(
            id="aluminum",
            label="Aluminum / Metal",
            default_finish="metallic",
            compatible_finishes=["metallic", "matte", "glossy"],
            description="Brushed or polished aluminum with authentic metallic sheen.",
        ),
        MaterialItem(
            id="kraft_paper",
            label="Kraft Paper",
            default_finish="matte",
            compatible_finishes=["matte", "embossed"],
            description="Natural unbleached recycled fibrous paper for organic aesthetics.",
        ),
        MaterialItem(
            id="plastic",
            label="Polymer / Plastic",
            default_finish="glossy",
            compatible_finishes=["glossy", "matte", "frosted"],
            description="PET and HDPE polymers for lightweight consumer goods packaging.",
        ),
    ],
    finishes=[
        TaxonomyItem(
            id="matte",
            label="Matte Soft-Touch",
            description="Non-glare velvet finish with soft ambient diffusion.",
        ),
        TaxonomyItem(
            id="glossy",
            label="High Gloss UV Varnish",
            description="Mirror-like specular reflections and rich color depth.",
        ),
        TaxonomyItem(
            id="metallic",
            label="Metallic Sheen",
            description="Subtle metallic flakes producing light-reactive surface sheen.",
        ),
        TaxonomyItem(
            id="embossed",
            label="Embossed / Debossed Relief",
            description="Raised 3D tactile contours creating authentic surface shadows.",
        ),
        TaxonomyItem(
            id="foil_stamping",
            label="Gold / Silver Foil Stamping",
            description="Reflective hot-stamped foil accents for luxury branding.",
        ),
        TaxonomyItem(
            id="frosted",
            label="Frosted Translucent",
            description="Etched semi-opaque surface with diffused light scattering.",
        ),
    ],
    viewpoints=[
        TaxonomyItem(
            id="front",
            label="Front Eye-Level",
            description="Direct eye-level frontal perspective emphasizing main label and form.",
        ),
        TaxonomyItem(
            id="three_quarter",
            label="Three-Quarter (3/4)",
            description="Hero perspective revealing front label, side depth, and volume.",
        ),
        TaxonomyItem(
            id="isometric",
            label="Isometric 3D",
            description="Balanced axonometric projection showing multiple panels simultaneously.",
        ),
        TaxonomyItem(
            id="top_down",
            label="Top-Down Flat Lay",
            description="90-degree overhead perspective ideal for product arrangements.",
        ),
    ],
    backgrounds=[
        TaxonomyItem(
            id="studio_clean",
            label="Clean Studio Infinity",
            description="Neutral studio cyclorama with soft ground contact shadow.",
        ),
        TaxonomyItem(
            id="retail_shelf",
            label="Retail Store Display",
            description="Commercial supermarket or boutique shelf with warm ambient lighting.",
        ),
        TaxonomyItem(
            id="minimalist_gradient",
            label="Minimalist Gradient",
            description="Modern aesthetic dual-tone monochromatic backdrop.",
        ),
        TaxonomyItem(
            id="outdoor_natural",
            label="Outdoor Natural Ambient",
            description="Soft sunlight and organic nature background elements.",
        ),
    ],
    lighting_presets=[
        TaxonomyItem(
            id="studio_softbox",
            label="Dual Studio Softbox",
            description="Balanced commercial product lighting with subtle catchlights.",
        ),
        TaxonomyItem(
            id="dramatic_rim",
            label="Dramatic Rim Lighting",
            description="Bold edge highlights separating package silhouette from backdrop.",
        ),
        TaxonomyItem(
            id="natural_golden",
            label="Golden Hour Warmth",
            description="Late afternoon directional warm sunlight and soft long shadows.",
        ),
    ],
)


@router.get(
    "/taxonomy",
    response_model=TaxonomyResponse,
    summary="Get packaging design taxonomy",
    description="Returns available package types, materials, finishes, viewpoints, backgrounds, and lighting presets.",
)
def get_taxonomy() -> TaxonomyResponse:
    return TAXONOMY_DATA
