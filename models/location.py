from pydantic import BaseModel, Field


class Location(BaseModel):
    """
    Geographic context for Local SEO targeting.

    First-class model reusable across ArticleRequest, Website config,
    Ahrefs keyword targeting, and future regional reporting.

    Supports country-agnostic structure — not limited to US markets.
    """

    city: str = Field(description="Target city, e.g. 'Denver'.")
    state: str = Field(description="State or province, e.g. 'Colorado' or 'CO'.")
    country: str = Field(default="US", description="ISO 3166-1 alpha-2 country code.")
    zip_code: str | None = Field(
        default=None,
        description="ZIP or postal code for hyper-local targeting."
    )
    neighborhood: str | None = Field(
        default=None,
        description="Neighborhood, district, or zone within the city, e.g. 'LoDo'."
    )
