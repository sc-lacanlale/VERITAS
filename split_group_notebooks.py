"""Regenerate Wild and OpenForensics Full (notebooks-of/) group notebooks."""
from __future__ import annotations

from make_defacto_notebook import main as make_defacto
from make_wild_notebook import main as make_wild


def main():
    make_wild()
    make_defacto()


if __name__ == "__main__":
    main()
