import re

def toknizer(text: str) -> set[str]:
    return set(
        re.findall(r"\w+", text.casefold())
    )
