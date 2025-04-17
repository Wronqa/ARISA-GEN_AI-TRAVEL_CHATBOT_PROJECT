import json
import os
from langchain.tools import Tool

DATA_FILE_PATH = "data/travel_offers.json"

def _format_offer_details(offer: dict) -> str:
    """Formats the found offer dictionary into a readable string."""
    details = []
    details.append(f"**Offer Found: {offer.get('offerCode', 'N/A')}**")
    details.append(f"**Title:** {offer.get('offerTitle', 'N/A')}")
    details.append(f"**Destination:** {offer.get('destinationCountry', 'N/A')}")
    details.append(f"**Duration:** {offer.get('durationNights', 'N/A')} nights")
    details.append(f"**Price:** ${offer.get('pricePerPersonUSD', 'N/A')} USD per person")
    details.append(f"**Departure:** {offer.get('departureDate', 'N/A')}")
    details.append(f"\n**Summary:**\n{offer.get('summary', 'N/A')}")

    core_exp = offer.get('coreExperiences', [])
    if core_exp:
        details.append("\n**Core Experiences:**")
        for exp in core_exp:
            details.append(f"- {exp}")


    return "\n".join(details)


def _find_offer_by_code(offer_code: str) -> str:
    """
    Searches the travel_offers.json file for an offer matching the given offerCode.

    Args:
        offer_code: The exact code of the offer to search for (e.g., 'CUB-HAV26').

    Returns:
        A formatted string with the offer details if found, or a 'not found' message.
    """
    if not offer_code or not isinstance(offer_code, str):
        return "Error: Please provide a valid offer code (string) as input."

    offer_code = offer_code.strip() 

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root = os.path.dirname(base_dir) 

        file_path = os.path.join(project_root, DATA_FILE_PATH) 

        if not os.path.exists(file_path):
             file_path = DATA_FILE_PATH 

        print(f"Attempting to load offers from: {file_path}") 

        if not os.path.exists(file_path):
            return f"Error: Data file not found at expected location: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            offers_data = json.load(f)

        for offer in offers_data:
            if offer.get("offerCode") == offer_code:
                print(f"Offer found for code: {offer_code}")
                return _format_offer_details(offer)

        print(f"Offer not found for code: {offer_code}")
        return f"Sorry, no travel offer found with the code: '{offer_code}'."

    except FileNotFoundError:
        return f"Error: The data file '{file_path}' was not found."
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON from the file '{file_path}'. Please check its format."
    except Exception as e:
        print(f"An unexpected error occurred in _find_offer_by_code: {e}")
        return f"An unexpected error occurred while searching for the offer: {e}"


offer_search_tool = Tool(
    name="Travel Offer Search by Code",
    func=_find_offer_by_code,
    description="Useful for searching the local travel_offers.json file to find details about a specific travel offer when the user provides its unique offer code (e.g., 'CUB-HAV26', 'FIN-LAP26'). Input MUST be the exact offer code string."
)
