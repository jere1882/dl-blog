from pydantic import BaseModel, Field
from typing import List

class Character(BaseModel):
    character_name: str = Field(
        description="Name of the character/deity/person identified in the scene"
    )
    reasoning: str = Field(
        description="Explanation of how/why you identified this character (e.g., 'Osiris is wearing his characteristic staff and crown')"
    )
    description: str = Field(
        description="Description and interesting facts about this character/deity"
    )
    location: str = Field(
        description="Location of the character in the picture (e.g., 'far left', 'center', 'right side', etc.)"
    )

class EgyptianArtAnalysis(BaseModel):
    characters: List[Character] = Field(
        description="List of characters/deities/people identified in the scene. Use 'unknown' or 'unidentified' if unclear. Should be empty if there are no clear depiction of characters, or if the picture only shows text/symbols with no clear depiction of characters."
    )
    picture_location: str = Field(
        description="Your best guess as to where this picture could have been taken - specific Valley of the Kings tomb, temple wall, etc. Use speculative language unless very confident."
    )
    interesting_detail: str = Field(
        description="Highlight an interesting detail of the picture that would be fascinating to a viewer."
    )
    date: str = Field(
        description="Your best guess as to when this may have been produced. Give one of the major Egyptian periods like Old Kingdom, Middle Kingdom, or New Kingdom."
    )
    ancient_text_translation: str = Field(
        description="Attempt to translate any ancient Egyptian text, symbols, or hieroglyphs, or at least try to identify individual elements (e.g., cartouches with royal names or deity names). If unable to translate, speculate about what it could be saying."
    ) 