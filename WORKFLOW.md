# Lab code release workflow

How to release code for a new paper, in the standard lab format.

## 1. Create the repo
- Go to the lab organization -> `paper-template` -> "Use this template".
- Name the repo after the paper (short name or acronym, lowercase, hyphens).
- Set visibility to **Private** while the work is in progress.

## 2. While working
- Core method code goes in `src/`.
- Every figure/table in the paper gets ONE script in `experiments/`
  (`figure1.py`, `table2.py`, ...). Fix random seeds.
- Keep `requirements.txt` up to date; pin versions before release.
- Fill in every `{{PLACEHOLDER}}` in `README.md`.

## 3. At arXiv / acceptance time
- Flip the repo to **Public** (Settings -> Danger zone -> Change visibility).
- Fill in `docs/index.html` (title, authors, abstract, teaser, bibtex, links).
- Enable the project page: Settings -> Pages -> Deploy from branch ->
  `main`, folder `/docs`.
- Add the repo + project page links to the arXiv abstract and the lab website.

## 4. Checklist before going public
- [ ] Fresh clone + `pip install -r requirements.txt` works
- [ ] Every experiment script runs and reproduces its figure/table
- [ ] README has no remaining `{{PLACEHOLDER}}`
- [ ] Bibtex entry is correct
- [ ] Project page is live
