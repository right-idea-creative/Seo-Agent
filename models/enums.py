from enum import Enum


class ImageRole(str, Enum):
    FEATURED = "featured"
    HERO = "hero"
    INLINE = "inline"
    GALLERY = "gallery"
    INFOGRAPHIC = "infographic"


class ArticleStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ArticleLanguage(str, Enum):
    ES = "es"
    EN = "en"
    PT = "pt"
    FR = "fr"


class ArticleTone(str, Enum):
    PROFESIONAL = "profesional"
    CASUAL = "casual"
    ACADEMICO = "academico"
    PERSUASIVO = "persuasivo"
    INFORMATIVO = "informativo"
    NARRATIVO = "narrativo"


class PublishStatus(str, Enum):
    DRAFT = "draft"
    PUBLISH = "publish"
    FUTURE = "future"


class SEOPlugin(str, Enum):
    YOAST = "yoast"
    RANKMATH = "rankmath"
    NONE = "none"


