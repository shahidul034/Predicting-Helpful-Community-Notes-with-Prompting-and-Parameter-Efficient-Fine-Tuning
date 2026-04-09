"""
Download Community Notes data from Twitter/X.

The data is available at:
https://communitynotes.twitter.com/guide/en/under-the-hood/download-data

This script downloads the three main TSV files:
- notes-*.tsv.gz: All notes with text, classification, metadata
- ratings-*.tsv.gz: Individual helpfulness ratings
- noteStatusHistory-*.tsv.gz: Status labels over time
"""

import os
import sys
import gzip
import argparse
from pathlib import Path
import aiohttp
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logging

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Download Community Notes data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Generate sample data for testing (useful when real data unavailable)",
    )
    return parser.parse_args()


async def download_file(
    session: aiohttp.ClientSession, url: str, filepath: Path
) -> bool:
    """Download a file asynchronously."""
    try:
        logger.info(f"Downloading: {url}")
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Failed to download {url}: HTTP {response.status}")
                return False

            total_size = int(response.headers.get("content-length", 0))
            logger.info(f"Size: {total_size / (1024 * 1024):.2f} MB")

            with open(filepath, "wb") as f:
                downloaded = 0
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"\rProgress: {progress:.1f}%")

        logger.info(f"\nSaved: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


async def download_all_files(output_dir: Path, overwrite: bool = False) -> None:
    """Download all Community Notes data files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base URL for Community Notes data (official Twitter/X bucket)
    # Note: URLs may change; check https://communitynotes.twitter.com/guide/en/under-the-hood/download-data
    base_url = "https://storage.googleapis.com/communitynotes-public-datasets"

    # Files to download (using the latest available version)
    files = {
        "notes": f"{base_url}/notes-00000-of-00001.tsv.gz",
        "ratings": f"{base_url}/ratings-00000-of-00001.tsv.gz",
        "note_status_history": f"{base_url}/noteStatusHistory-00000-of-00001.tsv.gz",
    }

    async with aiohttp.ClientSession() as session:
        tasks = []
        for name, url in files.items():
            filepath = output_dir / f"{name}.tsv.gz"
            if filepath.exists() and not overwrite:
                logger.info(f"Skipping {name} (already exists)")
                continue
            tasks.append(download_file(session, url, filepath))

        results = await asyncio.gather(*tasks)

    success_count = sum(results)
    logger.info(f"\nDownload complete: {success_count}/{len(files)} files downloaded")


def generate_sample_data(output_dir: Path, n_samples: int = 1000) -> None:
    """Generate sample data for testing when real data is unavailable."""
    import random
    import gzip
    from datetime import datetime, timedelta

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {n_samples} sample notes...")

    # Sample helpful notes
    helpful_notes = [
        "According to the CDC, over 95% of Americans are now fully vaccinated against COVID-19.",
        "The Joint Committee on Taxation reports that this bill will not increase taxes for 90% of Americans.",
        "NASA confirms that the James Webb Space Telescope has successfully captured images of distant galaxies.",
        "Multiple peer-reviewed studies have found no link between vaccines and autism.",
        "According to the IPCC, global temperatures have risen by 1.1°C since pre-industrial times.",
        "The World Health Organization states that hand washing for 20 seconds can prevent the spread of viruses.",
        "Data from the Bureau of Labor Statistics shows unemployment at a 50-year low.",
        "According to peer-reviewed research published in Nature, this claim is not supported by evidence.",
        "The FBI confirms that there is no evidence of widespread voter fraud in the 2020 election.",
        "Multiple fact-checking organizations have rated this claim as false based on available evidence.",
    ]

    # Sample unhelpful notes
    unhelpful_notes = [
        "This is obviously fake news and anyone who believes it is stupid.",
        "I don't like this person so their tweet is probably wrong.",
        "Everyone knows this is true, why are you even questioning it?",
        "This is just the government trying to control us.",
        "I heard on the radio that this is not true but I can't find a source.",
        "This is garbage and should be deleted immediately.",
        "Only idiots would believe this kind of nonsense.",
        "The media is lying about this, trust me.",
        "This is clearly biased and not worth reading.",
        "I disagree with this completely but have no evidence.",
    ]

    classifications = ["MISINFORMED_OR_POTENTIALLY_MISLEADING", "NOT_MISLEADING"]
    statuses = [
        "CURRENTLY_RATED_HELPFUL",
        "CURRENTLY_RATED_NOT_HELPFUL",
        "NEEDS_MORE_RATINGS",
    ]

    notes_data = []
    ratings_data = []

    base_date = datetime(2024, 1, 1)

    for i in range(n_samples):
        note_id = f"note_{i:08d}"
        author_id = f"user_{random.randint(10000, 99999)}"
        created_at = base_date + timedelta(
            days=random.randint(0, 364)
        )  # Full year 2024

        # 60% helpful, 40% not helpful
        if random.random() < 0.6:
            summary = random.choice(helpful_notes)
            status = "CURRENTLY_RATED_HELPFUL"
        else:
            summary = random.choice(unhelpful_notes)
            status = "CURRENTLY_RATED_NOT_HELPFUL"

        classification = random.choice(classifications)

        notes_data.append(
            {
                "noteId": note_id,
                "authorId": author_id,
                "summary": summary,
                "classification": classification,
                "currentLabelStatus": status,
                "createdAt": created_at.isoformat(),
                "ratingsCount": random.randint(5, 100),
            }
        )

        # Generate some ratings
        for j in range(random.randint(5, 20)):
            ratings_data.append(
                {
                    "noteId": note_id,
                    "contributorId": f"rater_{random.randint(10000, 99999)}",
                    "helpfulness": random.choice(["HELPFUL", "NOT_HELPFUL"]),
                    "createdAt": (
                        created_at + timedelta(hours=random.randint(1, 48))
                    ).isoformat(),
                }
            )

    # Write notes TSV
    notes_file = output_dir / "notes.tsv.gz"
    with gzip.open(notes_file, "wt", encoding="utf-8") as f:
        f.write("\t".join(notes_data[0].keys()) + "\n")
        for row in notes_data:
            f.write("\t".join(str(row[k]) for k in notes_data[0].keys()) + "\n")
    logger.info(f"Saved {len(notes_data)} notes to {notes_file}")

    # Write ratings TSV
    ratings_file = output_dir / "ratings.tsv.gz"
    with gzip.open(ratings_file, "wt", encoding="utf-8") as f:
        f.write("\t".join(ratings_data[0].keys()) + "\n")
        for row in ratings_data:
            f.write("\t".join(str(row[k]) for k in ratings_data[0].keys()) + "\n")
    logger.info(f"Saved {len(ratings_data)} ratings to {ratings_file}")

    # Write status history TSV
    status_file = output_dir / "noteStatusHistory.tsv.gz"
    with gzip.open(status_file, "wt", encoding="utf-8") as f:
        f.write("noteId\tstatus\tcreatedAt\n")
        for row in notes_data:
            f.write(
                f"{row['noteId']}\t{row['currentLabelStatus']}\t{row['createdAt']}\n"
            )
    logger.info(f"Saved status history for {len(notes_data)} notes to {status_file}")

    logger.info("Sample data generation complete!")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.sample_data:
        logger.info("Generating sample data for testing...")
        generate_sample_data(output_dir)
        return

    logger.info("Starting Community Notes data download...")
    logger.info(f"Output directory: {output_dir}")

    asyncio.run(download_all_files(output_dir, args.overwrite))

    logger.info("Download complete!")


if __name__ == "__main__":
    main()
