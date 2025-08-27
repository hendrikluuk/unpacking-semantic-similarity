#!/usr/bin/env python3
"""
  A script to search for matching PubMed abstracts based on a keyword query, get the corresponding PubMed IDs and download the top {k} abstracts for a list of (PMIDs).

  install BioPython if not already installed:
  pip install biopython  
"""
import os
import time

from Bio import Entrez
from Bio import Medline

email_address = os.getenv('PUBMED_ACCOUNT_EMAIL')

def search_pubmed(query, retmax=100):
    """
    Searches PubMed for a given query and returns a list of PMIDs.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=str(retmax))
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_abstracts(pmids, max_abstracts=10):
    """
    Fetches abstracts for a list of PMIDs.
    Returns a list of dictionaries, where each dictionary contains 'PMID' and 'Abstract'.
    """
    abstracts_data = []
    # Fetch in batches to be nice to the server
    batch_size = 200 # NCBI recommends fetching in batches of < 500
    pmids_to_fetch = pmids[:max_abstracts]

    for i in range(0, len(pmids_to_fetch), batch_size):
        batch_pmids = pmids_to_fetch[i:i+batch_size]
        try:
            handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="medline", retmode="text")
            records = Medline.parse(handle)
            for record in records:
                pmid = record.get("PMID", "")
                abstract = record.get("AB", "")
                if pmid and abstract:
                    abstracts_data.append({"PMID": pmid, "Abstract": abstract})
            handle.close()
        except Exception as e:
            print(f"Error fetching batch {i//batch_size + 1}: {e}")
        # Be respectful to NCBI servers
        time.sleep(0.34) # Limit requests to 3 per second without an API key

    return abstracts_data

def save_abstracts_to_file(abstracts_data, filename="pubmed_abstracts.txt"):
    """
    Saves the fetched abstracts to a text file.
    Each abstract is preceded by its PMID.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for item in abstracts_data:
            f.write(f"PMID: {item['PMID']}\n")
            f.write(f"Abstract: {item['Abstract']}\n\n")
    print(f"Successfully saved {len(abstracts_data)} abstracts to {filename}")


def main(query: str = "disease target gene therapeutic", k: int = 10000):
    if k < 1:
        raise ValueError("Number of abstracts must be positive.")

    if not email_address:
        print("Email address is required. Exiting.")
        exit()
    Entrez.email = email_address

    print(f"\nSearching PubMed for: \"{query}\"")
    try:
        pmids = search_pubmed(query, retmax=k*2) # Fetch more PMIDs initially in case some don't have abstracts
        if not pmids:
            print("No PMIDs found for your query.")
        else:
            print(f"Found {len(pmids)} potential PMIDs. Attempting to download top {k} abstracts...")
            abstracts = fetch_abstracts(pmids, max_abstracts=k)
            if abstracts:
                output_filename = f"out/pubmed_{query.replace(' ', '_')}_top_{k}.txt"
                save_abstracts_to_file(abstracts, output_filename)
            else:
                print("Could not retrieve any abstracts for the found PMIDs.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an internet connection and have provided a valid email.")

    print("\nScript finished.")

if __name__ == "__main__":
    main()
