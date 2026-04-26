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
    """Generate a challenging synthetic dataset with overlapping features to prevent trivial classification."""
    import random
    import gzip
    from datetime import datetime, timedelta

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {n_samples} challenging sample notes...")

    random.seed(42)

    subjects = [
        "climate change", "vaccines", "elections", "tax policy", "housing prices",
        "inflation rates", "unemployment", "education reforms", "healthcare costs",
        "renewable energy", "artificial intelligence regulation", "food safety",
        "water quality", "air pollution", "public transport", "immigration",
        "trade agreements", "cryptocurrency regulation", "data privacy", "cybersecurity"
    ]

    sources = [
        "the EPA reports", "a study in JAMA found", "according to the CDC",
        "the Department of Labor states", "researchers at MIT found",
        "the Federal Reserve noted", "a University of Michigan study shows",
        "NASA data indicates", "the WHO confirms", "Harvard researchers published"
    ]

    numbers = [
        "42%", "73%", "1.2 million", "5.6 million", "$23 billion", "$4.7 trillion",
        "18,000", "3.8 million", "67%", "91%", "0.4%", "23 million", "1.9 billion"
    ]

    directions = [
        "increased by", "decreased by", "remained at", "grew to", "fell below",
        "stayed above"
    ]

    timeframes = [
        "over the past year", "since 2019", "in Q3 2024", "as of March 2024",
        "between January and June", "during the first half of 2024",
        "for the fiscal year ending December", "between 2020 and 2024"
    ]

    neutral_verbs = [
        "stated", "reported", "finds", "indicates", "suggests", "notes", "observed"
    ]

    confident_verbs = [
        "proves", "confirms", "establishes", "demonstrates decisively", "leaves no doubt"
    ]

    hedges = [
        "may", "could", "potentially", "appears to", "tends to", "is likely to"
    ]

    qualifiers = [
        "the data suggests", "according to the available evidence", "based on recent figures",
        "as reported by", "research indicates", "findings show"
    ]

    filler_phrases = [
        "It is worth noting that", "In addition, it should be mentioned that",
        "Furthermore, one might consider that", "The point being made here is that",
        "On the other hand, there are those who argue that", "Some people would say that",
        "It could be argued that", "The general consensus seems to be that"
    ]

    subjective_phrases = [
        "clearly", "obviously", "undeniably", "it is absurd to think",
        "no one with common sense", "everyone knows", "it goes without saying"
    ]

    mild_subjective = [
        "I think", "in my opinion", "it seems to me", "personally I believe",
        "from my perspective", "I would argue"
    ]

    weak_sources = [
        "a blog post", "someone on social media", "I read somewhere", "a friend told me",
        "various internet sources", "according to some websites"
    ]

    # Generate notes with mixed signals
    def generate_confused_note():
        """Note that mixes helpful and unhelpful signals."""
        subject = random.choice(subjects)
        if random.random() < 0.5:
            parts = [
                f"{random.choice(filler_phrases)}",
                f"{random.choice(subjective_phrases)}",
            ]
            parts.append(f"the claim about {subject} {random.choice(hedges)} accurate")
            parts.append(f"though {random.choice(weak_sources)} suggests otherwise")
        else:
            parts = [
                f"Regarding {subject},",
                f"{random.choice(weak_sources)} indicates that",
                f"{random.choice(subjective_phrases)} this is the case, although I cannot verify this claim"
            ]
        return " ".join(parts)

    def generate_mildly_helpful_note():
        """Note with some evidence but weak reasoning."""
        subject = random.choice(subjects)
        source = random.choice(sources)
        number = random.choice(numbers)
        direction = random.choice(directions)
        timeframe = random.choice(timeframes)
        return (
            f"According to {source}, {subject} {direction} {number} {timeframe}. "
            "However, the sample size in the study was limited and results may vary."
        )

    def generate_helpful_note():
        """Well-structured note with clear evidence."""
        subject = random.choice(subjects)
        source = random.choice(sources)
        number = random.choice(numbers)
        direction = random.choice(directions)
        timeframe = random.choice(timeframes)
        qualifier = random.choice(qualifiers)
        return (
            f"The claim about {subject} is misleading. {source}. Specifically, {subject} "
            f"{direction} {number} {timeframe}. {qualifier}, the original tweet's figures "
            f"are not consistent with publicly available data from authoritative sources."
        )

    def generate_mixed_quality_note():
        """Note that is borderline helpful."""
        subject = random.choice(subjects)
        source = random.choice(sources)
        number = random.choice(numbers)
        direction = random.choice(directions)
        return (
            f"It appears that {subject} {direction} {number}. "
            f"While {source}, it's hard to say for certain whether this fully addresses the claim "
            f"in the original tweet. There may be other factors at play here."
        )

    def generate_noisy_helpful_note():
        """Helpful note with unnecessary noise."""
        subject = random.choice(subjects)
        source = random.choice(sources)
        number = random.choice(numbers)
        direction = random.choice(directions)
        filler = random.choice(filler_phrases)
        return (
            f"{filler} the statement about {subject} requires clarification. "
            f"{source}. The data shows {subject} {direction} {number}. "
            f"Some people might disagree with this interpretation, but the evidence "
            f"seems fairly conclusive on this particular matter at this point in time."
        )

    def generate_agnostic_note():
        """Note that doesn't take a clear stance."""
        subject = random.choice(subjects)
        return (
            f"The question of {subject} is complex. There are arguments on both sides "
            f"of this issue. Some experts believe one way, while others disagree. "
            f"More research is needed to reach a definitive conclusion."
        )

    def generate_misleading_note():
        """Note that cites a source but draws questionable conclusions."""
        subject = random.choice(subjects)
        source = random.choice(sources)
        number = random.choice(numbers)
        return (
            f"If you look at what {source}, you'll see {subject} increased by {number}. "
            f"This completely proves the original point in the tweet."
        )

    def generate_short_note():
        """Brief note with minimal information."""
        subject = random.choice(subjects)
        phrases = [
            f"The tweet about {subject} needs context.",
            f"Regarding {subject}: more information is needed.",
            f"This claim about {subject} is worth examining further.",
            f"Many people are confused about {subject} based on this post.",
        ]
        return random.choice(phrases)

    def generate_personal_note():
        """Note with personal experience that may or may not be helpful."""
        subject = random.choice(subjects)
        phrase = random.choice(mild_subjective)
        return (
            f"{phrase} the information about {subject} is inaccurate based on "
            f"what I've encountered in my own experience. I've worked in this field for "
            f"several years and the numbers don't seem to match the claim being made."
        )

    def generate_overconfident_correct_note():
        """Correct information but delivered in an overconfident manner."""
        subject = random.choice(subjects)
        source = random.choice(sources)
        number = random.choice(numbers)
        return (
            f"It's absolutely certain from what {source} that {subject} reached {number}. "
            f"Anyone who thinks otherwise is ignoring the data."
        )

    def generate_cautious_wrong_note():
        """Well-written but subtly flawed note."""
        subject = random.choice(subjects)
        source = random.choice(sources)
        return (
            f"It might be worth noting that, based on {source}, there appears to be some "
            f"confusion regarding {subject}. The author suggests figures could be different, "
            f"though this interpretation seems to conflate two separate datasets."
        )

    # Pool of generators with varying quality levels
    # Each generator produces notes with different helpfulness priors
    note_generators = {
        "clearly_helpful": (generate_helpful_note, 0.92),
        "mildly_helpful": (generate_mildly_helpful_note, 0.72),
        "noisy_helpful": (generate_noisy_helpful_note, 0.62),
        "mixed_quality": (generate_mixed_quality_note, 0.48),
        "overconfident_correct": (generate_overconfident_correct_note, 0.45),
        "agnostic": (generate_agnostic_note, 0.42),
        "personal_experience": (generate_personal_note, 0.38),
        "cautious_wrong": (generate_cautious_wrong_note, 0.32),
        "misleading": (generate_misleading_note, 0.22),
        "confused": (generate_confused_note, 0.15),
        "short_note": (generate_short_note, 0.50),
    }

    classifications = ["MISINFORMED_OR_POTENTIALLY_MISLEADING", "NOT_MISLEADING"]

    notes_data = []
    ratings_data = []

    base_date = datetime(2024, 1, 1)

    for i in range(n_samples):
        note_id = f"note_{i:08d}"
        author_id = f"user_{random.randint(10000, 99999)}"
        created_at = base_date + timedelta(days=random.randint(0, 364))

        # Pick a random generator
        gen_name, (gen_func, helpfulness_prior) = random.choice(
            list(note_generators.items())
        )

        summary = gen_func()

        # Add label noise: 12% of notes get flipped
        actual_helpfulness = random.random() + (helpfulness_prior - 0.5)
        if random.random() < 0.12:
            actual_helpfulness = 1.0 - actual_helpfulness

        status = (
            "CURRENTLY_RATED_HELPFUL"
            if actual_helpfulness > 0.5
            else "CURRENTLY_RATED_NOT_HELPFUL"
        )

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
