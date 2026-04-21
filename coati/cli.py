import argparse
import configparser
import logging
import os
import sys
from typing import Dict, List

from .graph.coati_graph import CoatiWordGraph
from .graph.takahe_graph import TakaheWordGraph
from .graph.reranker import KeyphraseReranker
from .scorer.grammar import GrammarScorer
from .utils.logger import setup_logging


def _load_clustered_sentences(filepath: str) -> Dict[str, List[str]]:
    """Load sentences grouped by cluster from a file.

    The file format uses lines starting with "classes_" as cluster
    separators. All subsequent lines until the next separator belong
    to that cluster.

    Args:
        filepath: Path to the input file.

    Returns:
        A dictionary mapping cluster identifiers to lists of sentence strings.
    """
    clustered_sentences: Dict[str, List[str]] = {}
    sentences: List[str] = []

    with open(filepath, "r") as text:
        for line in text:
            line = line.strip()
            if line.startswith("classes_"):
                sentences = []
                clustered_sentences[line] = sentences
            else:
                sentences.append(line)

    return clustered_sentences


def _save_results(
    results: Dict[str, List[str]], save_path: str, filename: str
) -> None:
    """Save compression results to a file.

    Args:
        results: A dictionary mapping cluster identifiers to result lines.
        save_path: Directory path where the output file will be written.
            Created automatically if it does not exist.
        filename: Name of the output file.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filepath = os.path.join(save_path, filename)
    logging.info("Saving file[%s]", filepath)

    with open(filepath, "w") as save_file:
        for key, lines in results.items():
            save_file.write(key + "\n")
            save_file.writelines(lines)
            save_file.flush()

    logging.info("Save file[%s] success!", filepath)


def run_takahe_compression(
    sentences_dir: str, save_dir: str, use_keyphrase: bool = False
) -> None:
    """Run Takahe (frequency-based) multi-sentence compression on a directory.

    Reads POS-tagged sentence files from the "tagged" subdirectory,
    compresses each cluster of sentences, and saves the results.

    Args:
        sentences_dir: Root directory containing a "tagged" subdirectory
            with input files.
        save_dir: Root directory for output files. Results are saved
            under "protogenesis" or "keyphrases" subdirectories.
        use_keyphrase: If True, apply keyphrase reranking to the
            compression candidates (default: False).
    """
    sub_dir = "tagged"
    for parent, _dirs, files in os.walk(os.path.join(sentences_dir, sub_dir)):
        for filename in files:
            logging.info("Compressing: %s", os.path.join(parent, filename))

            clustered_sentences = _load_clustered_sentences(
                os.path.join(parent, filename)
            )

            proto_results: Dict[str, List[str]] = {}
            kp_results: Dict[str, List[str]] = {}

            for key, sentences in clustered_sentences.items():
                if not use_keyphrase:
                    logging.info(
                        "[protogenesis]compressing, filename=%s, class=%s",
                        filename,
                        key,
                    )
                    compresser = TakaheWordGraph(sentences)
                    candidates = compresser.get_compression(50)

                    tmp = [(score / len(path), path) for score, path in candidates]
                    tmp.sort(key=lambda x: x[0])

                    proto_results[key] = [
                        f"{round(score, 6)}#{' '.join(u[0] for u in path)}\n"
                        for score, path in tmp
                    ]

                if use_keyphrase:
                    logging.info(
                        "[keyphrases]compressing, filename=%s, class=%s",
                        filename,
                        key,
                    )
                    compresser = TakaheWordGraph(sentences)
                    candidates = compresser.get_compression(50)

                    reranker = KeyphraseReranker(sentences, candidates, lang="en")
                    reranked = reranker.rerank_nbest_compressions()

                    kp_results[key] = [
                        f"{round(score, 6)}#{' '.join(u[0] for u in path)}\n"
                        for score, path in reranked
                    ]

            logging.info("Compress file[%s] finished!", filename)

            if not use_keyphrase:
                _save_results(proto_results, os.path.join(save_dir, "protogenesis"), filename)
            if use_keyphrase:
                _save_results(kp_results, os.path.join(save_dir, "keyphrases"), filename)


def run_event_compression(
    sentences_dir: str,
    save_dir: str,
    grammar_scorer: GrammarScorer,
    lambd: float,
    max_neighbors: int,
    queue_size: int,
    use_keyphrase: bool = False,
) -> None:
    """Run event-guided multi-sentence compression on a directory.

    Reads weighted sentence files from the "weighted" subdirectory,
    compresses each cluster using the event-guided algorithm with
    language model scoring, and saves the results.

    Args:
        sentences_dir: Root directory containing a "weighted" subdirectory
            with input files.
        save_dir: Root directory for output files. Results are saved
            under "events" or "events_keyphrases" subdirectories.
        grammar_scorer: A GrammarScorer instance for fluency evaluation.
        lambd: Weight balancing path score and fluency score.
        max_neighbors: Maximum number of successor neighbors in BFS.
        queue_size: Maximum BFS queue capacity.
        use_keyphrase: If True, apply keyphrase reranking (default: False).
    """
    sub_dir = "weighted"
    for parent, _dirs, files in os.walk(os.path.join(sentences_dir, sub_dir)):
        for filename in files:
            logging.info("Compressing: %s", os.path.join(parent, filename))

            clustered_sentences = _load_clustered_sentences(
                os.path.join(parent, filename)
            )

            event_results: Dict[str, List[str]] = {}

            for key, sentences in clustered_sentences.items():
                logging.info(
                    "[events]compressing, filename=%s, class=%s", filename, key
                )

                compresser = CoatiWordGraph(sentences, grammar_scorer)
                candidates = compresser.event_guided_multi_compress(
                    lambd, max_neighbors, queue_size, 50
                )

                if use_keyphrase:
                    ksp_candidates = compresser.get_compression(50)
                    reranker = KeyphraseReranker(sentences, ksp_candidates, lang="en")
                    reranked = reranker.rerank_nbest_compressions()
                    event_results[key] = [
                        f"{round(score, 6)}#{' '.join(u[0] for u in path)}\n"
                        for score, path in reranked
                    ]
                else:
                    event_results[key] = [
                        f"{round(score, 6)}#{sentence}\n"
                        for score, sentence in candidates
                    ]

                logging.info(
                    "[events]compress success, filename=%s, class=%s", filename, key
                )

            logging.info("Compress file[%s] finished!", filename)

            output_subdir = "events_keyphrases" if use_keyphrase else "events"
            _save_results(event_results, os.path.join(save_dir, output_subdir), filename)


def main() -> None:
    """Entry point for the Coati CLI.

    Supports three subcommands:
    - takahe: Run frequency-based compression.
    - event: Run event-guided compression with language model scoring.
    - config: Run with a legacy configuration file.
    """
    parser = argparse.ArgumentParser(
        description="Coati - Multi-sentence compression based on word graph algorithms"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    takahe_parser = subparsers.add_parser("takahe", help="Run takahe compression")
    takahe_parser.add_argument("--input", required=True, help="Input sentences directory")
    takahe_parser.add_argument("--output", required=True, help="Output directory")
    takahe_parser.add_argument(
        "--keyphrase", action="store_true", help="Use keyphrase reranking"
    )

    event_parser = subparsers.add_parser("event", help="Run event-guided compression")
    event_parser.add_argument("--input", required=True, help="Input sentences directory")
    event_parser.add_argument("--output", required=True, help="Output directory")
    event_parser.add_argument("--ngram-model", required=True, help="N-gram model path")
    event_parser.add_argument("--lambda", type=float, default=1.0, help="Path/LM score weight")
    event_parser.add_argument("--max-neighbors", type=int, default=6, help="Max successor neighbors")
    event_parser.add_argument("--queue-size", type=int, default=1024, help="BFS queue size")
    event_parser.add_argument(
        "--keyphrase", action="store_true", help="Use keyphrase reranking"
    )

    config_parser = subparsers.add_parser(
        "config", help="Run with configuration file (legacy mode)"
    )
    config_parser.add_argument("config_file", help="Configuration file path")

    args = parser.parse_args()

    setup_logging()

    if args.command == "takahe":
        run_takahe_compression(args.input, args.output, args.keyphrase)

    elif args.command == "event":
        logging.info("Initializing ngram model[%s]", args.ngram_model)
        grammar_scorer = GrammarScorer(args.ngram_model)
        run_event_compression(
            args.input,
            args.output,
            grammar_scorer,
            getattr(args, "lambda"),
            args.max_neighbors,
            args.queue_size,
            args.keyphrase,
        )

    elif args.command == "config":
        cf = configparser.ConfigParser()
        cf.read(args.config_file)

        sentences_dir = cf.get("emsc", "sentences_dir")
        save_dir = cf.get("emsc", "save_dir")
        ngram_modelpath = cf.get("emsc", "ngram_model_path")
        lambd = cf.getfloat("emsc", "lambd")
        max_neighbors = cf.getint("emsc", "max_neighbors")
        queue_size = cf.getint("emsc", "queue_size")

        logging.info("Initializing ngram model[%s]", ngram_modelpath)
        grammar_scorer = GrammarScorer(ngram_modelpath)
        run_event_compression(
            sentences_dir,
            save_dir,
            grammar_scorer,
            lambd,
            max_neighbors,
            queue_size,
        )

    else:
        parser.print_help()
        sys.exit(1)

    logging.info("program finish!")


if __name__ == "__main__":
    main()
