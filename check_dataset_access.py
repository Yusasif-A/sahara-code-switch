"""
Has Hugging Face approved the gated benchmark datasets yet?

The AfriSwitch collection is gated, and the interesting part is *how*:

    AfriSwitch      gated="manual"  a human at Intron approves each request,
                                    so it sits pending until they get to it
    AfriSwitchCare  gated="auto"    granted the moment you accept the terms
                                    on the dataset page

That distinction matters when you are waiting: manual gating is not something
you can hurry, but auto gating is a single click you may simply not have made.

This only asks the Hub about access, so it answers in a second or two rather
than downloading gigabytes to find out.

    python check_dataset_access.py
"""

from __future__ import annotations

import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from config import settings

DATASETS = [
    ("intronhealth/AfriSwitch", "16.6k code-switched clips — the main benchmark set"),
    ("intronhealth/AfriSwitchCare", "108 medical clips, 10 languages"),
]


def main() -> None:
    token = settings.stt.hf_token or None
    if not token:
        print("\n  No HF_TOKEN in .env — add it before checking gated datasets.\n")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"\n  Token: {token[:6]}...{token[-4:]}\n")

    ready = []
    for name, description in DATASETS:
        try:
            # auth_check tests access to the repo *contents*, which is the
            # question. dataset_info only reads metadata, and that stays public
            # on a gated dataset — using it reported AfriSwitch as READY minutes
            # after load_dataset had been refused for exactly that repo.
            api.auth_check(name, repo_type="dataset")
            info = api.dataset_info(name)
        except GatedRepoError:
            print(f"  PENDING   {name}")
            print(f"            {description}")
            print(f"            Request access: https://huggingface.co/datasets/{name}\n")
            continue
        except RepositoryNotFoundError:
            print(f"  MISSING   {name} — renamed, removed, or not visible to this token\n")
            continue
        except Exception as exc:
            print(f"  ERROR     {name}: {str(exc).splitlines()[0][:90]}\n")
            continue

        gating = getattr(info, "gated", None)
        # dataset_info succeeding on a gated repo means the gate is open to us.
        print(f"  READY     {name}   (gating: {gating or 'none'})")
        print(f"            {description}\n")
        ready.append(name)

    if "intronhealth/AfriSwitch" in ready:
        print("  The main set is open. Pull real code-switched samples with:")
        print()
        print("      python benchmark.py --fetch-afriswitch 30")
        print("      python benchmark.py --report benchmark_results.json")
        print()
    elif "intronhealth/AfriSwitchCare" in ready:
        # Deliberately not suggesting --fetch-afriswitch here: it targets the
        # dataset that is still pending, and running it just reprints the same
        # refusal. Point at what is actually reachable instead.
        print("  AfriSwitch is still pending manual approval, but AfriSwitchCare")
        print("  is open and carries the four Nigerian languages. Use it now:")
        print()
        print("      python benchmark.py --fetch-afriswitchcare 20")
        print("      python benchmark.py --report benchmark_results.json")
        print()
    else:
        print("  Nothing available yet. AfriSwitch is manually approved, so this is")
        print("  a wait rather than a misconfiguration. The synthesised set still")
        print("  exercises the whole harness:")
        print()
        print("      python benchmark.py --make-samples")
        print("      python benchmark.py --report benchmark_results.json")
        print()


if __name__ == "__main__":
    main()
