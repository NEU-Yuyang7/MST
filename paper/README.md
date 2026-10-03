# BMSBoruvka.tex

Converted from the original Word draft. Structure, numbering, and content
should match the .docx closely:

- Proper \section / \subsection commands (not bold-text headings)
- \begin{theorem}/\begin{lemma}/\begin{remark}/\begin{proof} environments
  (amsthm), auto-numbered to match the plain-text cross-references already
  in the prose ("Theorem 2", "Lemma 1", etc.)
- All 9 tables converted to longtable and verified to fit within margins
- Reference list cleaned into a hanging-indent style, DOIs as clickable links

## Compiling

**Use XeLaTeX (or LuaLaTeX), not pdflatex:**

    xelatex BMSBoruvka.tex   (run twice for cross-references)

The source uses Unicode characters directly (Borůvka's "ů", ξ, θ, and some
math glyphs) rather than LaTeX escape sequences, so it relies on fontspec +
a system Unicode font (DejaVu Serif/Math, set in the preamble) rather than
pdfTeX's 8-bit font model. Both xelatex and lualatex are on essentially
every modern TeX distribution (TeX Live, MiKTeX, Overleaf all default to
offering these engines), so this shouldn't require installing anything
extra, but if you paste this into Overleaf, double-check the project's
compiler setting (menu: Compiler) is set to XeLaTeX before building.

The enclosed BMSBoruvka.pdf is the actual compiled output (verified: clean
compile, no errors, 17 pages), so you can confirm visually that nothing
was lost before diffing against the source .tex.

## What to check

I converted this mechanically (regex-based transformation from pandoc's
docx output, not retyped by hand), so please proofread, particularly:

- The abstract and section bodies for any inline-bold phrases that should
  have stayed as emphasis but might have been swept up by the heading
  logic (I checked for this but a second pass from you/your student is
  worth doing).
- Theorem/Lemma names in brackets -- e.g. \begin{lemma}[Phase B, Sparse
  Graphs] -- came from parsing the original text's parenthetical names,
  which is usually clean but worth a glance.
- Citation keys ([Bor26], [PR02], etc.) are plain bracketed text, not
  \cite{} commands backed by a .bib file -- fine for a draft, but you may
  want proper BibTeX/biblatex before camera-ready submission.
