import argparse
import sys

# Import core detection modules
from src.hashing import generate_phash
from src.embedding import generate_embeddings, load_model
from src.search import build_faiss_index, search_duplicates

def cli_index(args):
    """Handles the 'index' sub-command."""
    print(f"Loading model: {args.model_name}...")
    model = load_model(args.model_name)
    
    print(f"Generating embeddings and hashes from {args.source_dir}...")
    metadata, embeddings = generate_embeddings(args.source_dir, model, recursive=args.recursive, batch_size=args.batch_size)
    
    print(f"Building FAISS index at {args.db_path}...")
    build_faiss_index(metadata, embeddings, args.db_path)
    print("Indexing complete.")

def cli_search(args):
    """Handles the 'search' sub-command."""
    if not args.db_path:
        print("Error: --db-path is required for searching.")
        sys.exit(1)

    print(f"Loading database from {args.db_path}...")
    index, db_metadata = search_duplicates(args.db_path) # Simplified function call
    
    # ... (Logic to process query image/dir and search index) ...
    
    print(f"Searching query {args.query} against database...")
    # This function would contain the multi-stage logic
    results = search_duplicates(args.query, index, db_metadata, args.threshold, args.stage_3_verify, args.max_results)
    
    print(f"Found {len(results)} potential duplicate pairs.")
    # (Logic to save results to --output-file)

def main():
    parser = argparse.ArgumentParser(description="Highly Sophisticated Duplicate AI Image Detector.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Index Sub-command ---
    index_parser = subparsers.add_parser('index', help='Process images and build the searchable index.')
    index_parser.add_argument('-s', '--source-dir', required=True, help='Root directory containing images to index.')
    index_parser.add_argument('-d', '--db-path', required=True, help='File path to save the output FAISS index/metadata.')
    index_parser.add_argument('-m', '--model-name', default='resnet50', help='Deep learning model name (e.g., resnet50, vit-base).')
    index_parser.add_argument('-r', '--recursive', action='store_true', help='Scan sub-directories recursively.')
    index_parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch size for GPU processing.')
    index_parser.set_defaults(func=cli_index)

    # --- Search Sub-command ---
    search_parser = subparsers.add_parser('search', help='Search for duplicates against an existing index.')
    search_parser.add_argument('-q', '--query', required=True, help='Path to a single image or a directory of images to check.')
    search_parser.add_argument('-d', '--db-path', required=True, help='File path to the pre-built FAISS index to search against.')
    search_parser.add_argument('-t', '--threshold', type=float, default=0.90, help='Cosine Similarity score cutoff (0.0 to 1.0).')
    search_parser.add_argument('-o', '--output-file', help='Saves the results to a JSON or CSV file.')
    search_parser.add_argument('-v', '--stage-3-verify', action='store_true', help='Activates SIFT/ORB verification for borderline matches.')
    search_parser.add_argument('-k', '--max-results', type=int, default=5, help='Max nearest neighbors to retrieve from the index.')
    search_parser.set_defaults(func=cli_search)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
  
