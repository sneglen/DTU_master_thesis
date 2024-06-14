# standard libraries
import logging

logger = logging.getLogger(__name__)


def get_instructions(key: str, article_text: str, quote_text: str, per_dict: dict, org_dict: dict, src_dict: dict) -> str:

  src_entity = ""
  org_entities = ""

  instructions = None
  # get per_entities and src_entities for QUOTE annotation instructions (SRC, GEN, FUN, ROL, EMP)
  if key not in  ["PER", "ORG", "QUOTES"]:
    if per_dict is not None:
      per_entities = ', '.join(per_dict["PER"])
    if org_dict is not None:
      org_entities = ', '.join(org_dict["ORG"])
    if src_dict is not None and src_dict["SRC"] != [] and src_dict["SRC"] != ['']:
      src_entity = src_dict["SRC"][0] # only one SRC entity

  # Quoted SRC entity
  if len(src_entity) > 0:
    quoted_src_entity_str = f"Quoted person:\n{src_entity}\n"
  else:
    quoted_src_entity_str = ""

  # Quoted ORG entity
  if len(org_entities) > 0:
    quoted_org_entity_str = f"Quoted organization:\n{org_entities}\n"
  else:
    quoted_org_entity_str = ""


## PER --------------------------------------------------------------------------------------
  if key == "PER":
    instructions = (
      f'Identify and extract in Danish all the unique persons, also children, from the article.\n' 
      f'You are not allowed to create your own strings. All strings you return must be extracted from the article in Danish.\n'
      f'Article:\n{article_text}'
    )


## ORG --------------------------------------------------------------------------------------
  elif key == "ORG":
    instructions = (
      f'You are a helpful and expert annotation assistant.\n'
      f'Identify and extract in Danish all the unique organizations and companies from the article.\n' 
      f'You are not allowed to create your own strings. All strings you return must be extracted from the article in Danish.\n'
      f'Article:\n{article_text}'
    )

## QUOTES -----------------------------------------------------------------------------------
  elif key == "QUOTES":
    instructions =  (
      f'Read the following Danish article and extract direct quotations.\n'
      f'Ensure each quotation is accurately captured and completely separated from the surrounding narrative text.\n'
      f'Return the quotations as a list.\n'
      f'Article:\n{article_text}'
    )

## SRC --------------------------------------------------------------------------------------
  elif key == "SRC":
    if len(per_dict["PER"]) > 0:
      # IDEAL case
      instructions = ( 
        f'You are a helpful and expert annotation assistant.\n'
        f'Identify and extract the source of the quotation in the article.\n'
        f'The source must be one and only one person from the list of persons\n'
        f'If no match is found, return an empty string.\n'
        f'List of persons:\n{per_entities}.\n'
        f'Quotation:\n{quote_text}\n'
        f'Article:\n{article_text}'
      )

    else:
      logger.notice(f"{key} annotation error: SRC should not have been invoked when PER_dict is empty. Returning 'None' instructions.")
      instructions = None

## GEN --------------------------------------------------------------------------------------
  elif key == "GEN":
    if len(src_dict["SRC"]) > 0 and src_dict["SRC"] != ['']:
      instructions = (
        f'Determine the gender of the quoted person: {src_entity}.\n' 
        f'"M" for male, "F" for female, "X" for unknown or not possible to determine.\n'
        f'Quotation:\n{quote_text}\n'
        f'Article:\n{article_text}'
        )
    else:
      instructions = (
        f'Determine the gender of the quoted entity if applicable.\n' 
        f'"M" for male, "F" for female, "X" for unknown or not possible to determine.\n'
        f'Quotation:\n{quote_text}\n'
        f'Article:\n{article_text}'
        )

## FUN --------------------------------------------------------------------------------------
  elif key == "FUN":
    instructions = (
      f'You are a helpful and expert annotation assistant.\n'
      f'Determine the role of the quoted person in the context of the article. Return only one of these eight types:\n"Expert", "Case", "Politician", "DR source", "Interest organization", "Professional expert", "Authority", "Other".\n' 
      # Simple English
      f'"Expert": Person with high knowledge, such as a university professor.\n' 
      #f'Example for "Expert": "It looks depressing, says Peter Jensen, who is a professor at the Technical University of Denmark, DTU.". In this example "Peter Jensen" is an "Expert". '
      f'"Case": Person with telling about personal experiences.\n'
      #f'Example for "Case": "Michael, who is a family man, was not happy with the cuts at the institutions.". Here "Michael" would be the "Case" DR selected to interview. A hypothetical mentioned "hobbyist" talking about his LEGO collection could also be an example of "Case". '
      f'"Politician": Person working in politics.\n'
      #f'Example for "Politician": "Foreign Minister Jakob Ellemann-Jensen decided to visit NATO". In this example "Jakob Ellemann-Jensen" is a "Politician". '
      f'"DR source": Person directly associated with DR, such as DR employees or correspondents.\n'
      #f'Example for "DR source": "I could use another cars, says Holger Sandberg...". If there are no other references of who "Holger Sandberg" is or who he works for, then "Holger Sandberg" is likely a DR source. If "Holger" is a "DR korrespondent", then there is no doubt "Holger" is a "DR source". '
      f'"Interest organization": Organizations like working unions or business associations.\n'
      #f'Example for "Interest organization": Could e.g. trade unions like "DJØF" or "Dansk Industri". '
      f'"Professional expert": Person with high professional/industry knowledge.\n'
      #f'Example for "Professional expert": "It is problematic, says Mogens Fabricius, who is technical advisor at COWI". In this example "Mogen Fabricius" is a "Professional expert". ' 
      f'"Authority": Official bodies or institutions like the police or health authority.\n'
      #f'Example for "Authority": "Danish Tax Ministry" and "Police" are good examples. '
      f'"Other": Use this category when none of the other categories apply.\n'
      #f'Example for "Other": This type MUST be used if no other function type is suitable. '
      f'{quoted_src_entity_str}'
      f'Quotation:\n{quote_text}\n'
      f'Article:\n{article_text}'
    )

## ROL --------------------------------------------------------------------------------------
  elif key == "ROL":
    instructions = (
      f'You are a helpful and expert annotation assistant.\n'
      f'Identify and extract the role or function in Danish of the quoted entity of the article.\n'
      f'You are not allowed to create your own strings. All strings you return must be extracted from the article in Danish.\n'
      f'Usually the role of the quoted entity is a person with a role, e.g. "employed", but if the person is "boss" and "professor" then the roles are ["boss", "professor"].\n'
      f'{quoted_src_entity_str}'
      f'Quotation:\n{quote_text}\n'
      f'Article:\n{article_text}'
    )

## EMP --------------------------------------------------------------------------------------
  elif key == "EMP":
    instructions = (
      f'You are a helpful and expert annotation assistant.\n'
      f'Identify and extract the employer (EMP) of the quoted entity in the Danish article. The employer is typically an organization or company.\n'
      f'You are not allowed to create your own strings. All strings you return must be extracted from the article in Danish.\n'
      f'If the employer field should be empty, return nothing. If you find a valid employer in the article, fetch and return this string in Danish from the article.'
      f'{quoted_src_entity_str}'
      f'{quoted_org_entity_str}'
      f'Quotation:\n{quote_text}\n'
      f'Article:\n{article_text}'
    )

  else:
    logger.error(f"{key} annotation error: Cannot fetch instructions without for unknown key. Returning 'None' instructions.")
    return None

  return instructions