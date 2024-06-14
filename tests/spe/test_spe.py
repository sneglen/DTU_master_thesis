# local libraries
from src.spe.agents.annotator_agents import parse_json_with_fallback

## Testing valid JSON string: Article 30
def test_valid_json_article_30(logger, request):
    logger.info(f"TEST: {request.node.name}")

    input_response = '{"PER" : ["Camilla Valetas", "Elliot", "Torben Foged", "Helen Amundsen"]}'
    expected = ["Camilla Valetas", "Elliot", "Torben Foged", "Helen Amundsen"]
    
    res = parse_json_with_fallback(input_response)
    (dict, decoding_ok) = res
    
    assert decoding_ok, "Decoding should have succeeded for valid JSON"
    assert dict["PER"] == expected, "Incorrect PER list for Article 30"

## Testing valid JSON string: Article 0
def test_valid_json_article_0(logger, request):
    logger.info(f"TEST: {request.node.name}")

    input_response = '{"PER": ["Alison Van Uytvanck", "Caroline Wozniacki", "Johanna Larsson"]}'
    expected = ["Alison Van Uytvanck", "Caroline Wozniacki", "Johanna Larsson"]
    
    res = parse_json_with_fallback(input_response)
    (dict, decoding_ok) = res
    
    assert decoding_ok, "Decoding should have succeeded for valid JSON"
    assert dict["PER"] == expected, "Incorrect PER list for Article 0"
 
 
## Testing invalid JSON string: Supposed Article 0 mistake 
def test_invalid_json(logger, request):
    logger.info(f"TEST: {request.node.name}")

    input_response = '{"PER": ["Alison Van Uytvanck" "Caroline Wozniacki", "Johanna Larsson"]}'  
    # Missing comma                                 ***
    res = parse_json_with_fallback(input_response)

    (dict, decoding_ok) = res

    assert not decoding_ok, "Decoding expected to fail for invalid JSON"
    assert dict is None, "dict expected to be 'None' when decoding fails"
