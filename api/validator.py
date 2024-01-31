from pydantic import BaseModel, Field, confloat

class WineInput(BaseModel):
    type: str = Field(..., pattern='^(red|white)$')
    fixed_acidity: confloat(ge=0.0, le=15.0)
    volatile_acidity: confloat(ge=0.0, le=2.0)
    citric_acid: confloat(ge=0.0, le=1.0)
    residual_sugar: confloat(ge=0.0, le=50.0)
    chlorides: confloat(ge=0.0, le=0.5)
    free_sulfur_dioxide: confloat(ge=0.0, le=70.0)
    total_sulfur_dioxide: confloat(ge=0.0, le=300.0)
    density: confloat(ge=0.0, le=2.0)
    pH: confloat(ge=0.0, le=4.0)
    sulphates: confloat(ge=0.0, le=2.0)
    alcohol: confloat(ge=0.0, le=20.0)
