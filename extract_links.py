import csv, re, requests, sys
from tqdm import tqdm
from collections import Counter


def clean_url(url):
    embedded_url_match = re.search(r'\((http[s]?://\S+)\)|\[(http[s]?://\S+)\]', url)
    if embedded_url_match:
        url = embedded_url_match.group(1) or embedded_url_match.group(2)
    url = re.sub(r'[\]\)\]]+$', '', url)
    url = url.strip().replace("*", "")
    url = url.removesuffix(".")
    url = url.removesuffix(",")
    url = re.sub(r'[\]\)\]]+$', '', url)

    return url

def get_final_url(short_url):
    short_url = clean_url(short_url)

    try:
        response = requests.get(short_url, allow_redirects=True, timeout=20)
        return response.url
    except Exception as e:
        return None

def extract_links(input_file):
    output_file = f"{input_file.removesuffix('.csv')}_links.csv"
    output_link_csv = f"{input_file.removesuffix('.csv')}_link_counts.csv"
    
    with open(input_file, mode='r', encoding='utf-8') as file:
        total_rows = sum(1 for _ in csv.DictReader(file))

    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['Links']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        all_links = []
        error_links = []
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        for row in tqdm(reader, total=total_rows):
            body_text = row.get('Body', '')
            links = url_pattern.findall(body_text)
            cleaned_links = []
            
            if links:
                for link in links:
                    final_url = get_final_url(link)
                    if final_url is None:
                        error_links.append(link)
                    else:
                        final_url = final_url.replace(":443", "")
                        cleaned_links.append(final_url)
                        all_links.append(final_url)
                
                row['Links'] = cleaned_links
                writer.writerow(row)
    
    domains = [re.match(r'https?://[^/]+', link).group() for link in all_links if link]
    domain_counts = Counter(domains)
    filtered_domain_counts = {domain: count for domain, count in domain_counts.items() if count > 0}
    sorted_domain_counts = dict(sorted(filtered_domain_counts.items(), key=lambda item: item[1], reverse=True))
    with open(output_link_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Domain", "Count"])
        for domain, count in sorted_domain_counts.items():
            writer.writerow([domain, count])

    print(f"Links extracted to {output_file}")

    print("Error links:")
    for link in error_links:
        print(link)


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <jsonl_file_1> <jsonl_file_2> ... <jsonl_file_n>")
        sys.exit(1)
    
    files = sys.argv[1:]
    
    for file in files:
        print(f"Extracting links from {file}")
        extract_links(file)
        print(f"Completed extracting links from {file}\n")


if __name__ == "__main__":
    main()