# Build the manuscript (main.pdf) and the Supplementary Information (supplement.pdf). Each resolves the other's
# labels through xr-hyper, so the supplement is built, then the manuscript, then the supplement again.
# Refresh numbers first:  cd .. && python analysis/results.py && python analysis/posthoc.py && python analysis/si.py && python analysis/figures.py
# (generated/replay.tex comes from python analysis/replay.py, which takes longer and needs bubblewrap; see README)
# Uses pdflatex + bibtex if installed, otherwise Tectonic. Without either:
#   conda create -n tex -c conda-forge tectonic   and then   conda run -n tex make
PDFLATEX := $(shell command -v pdflatex 2>/dev/null)
TECTONIC := $(shell command -v tectonic 2>/dev/null)
SOURCES := main.tex supplement.tex preamble.tex sections/*.tex refs.bib generated/* figures/*

all: main.pdf supplement.pdf

main.pdf supplement.pdf &: $(SOURCES)
ifneq ($(PDFLATEX),)
	for doc in supplement main supplement; do \
	  pdflatex -interaction=nonstopmode $$doc && bibtex $$doc && \
	  pdflatex -interaction=nonstopmode $$doc && pdflatex -interaction=nonstopmode $$doc || exit 1; \
	done
else ifneq ($(TECTONIC),)
	tectonic --keep-intermediates --keep-logs supplement.tex
	tectonic --keep-intermediates --keep-logs main.tex
	tectonic --keep-intermediates --keep-logs supplement.tex
else
	@echo "No LaTeX engine found. Either install TeX Live (sudo apt install texlive-latex-extra)"
	@echo "or Tectonic without sudo: conda create -n tex -c conda-forge tectonic, then: conda run -n tex make"
	@exit 1
endif

# No brace expansion here: make runs recipes with /bin/sh, which is dash on Debian and Ubuntu.
clean:
	rm -f $(foreach doc,main supplement,$(addprefix $(doc).,aux bbl blg log out toc xdv pdf))

.PHONY: all clean
