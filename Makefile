transformers-lecture.pdf: transformers-lecture.md $(wildcard img/*)
	pandoc                                                                     \
		--columns=50                                                       \
		--dpi=300                                                          \
		--listings                                                         \
		-M date="$(shell date '+%B %d, %Y')"                               \
		-o $@                                                              \
		--pdf-engine lualatex                                              \
		-s                                                                 \
		--shift-heading-level=0                                            \
		--slide-level 2                                                    \
		--template pandoc-beamer-how-to/pandoc/templates/default_mod.latex \
		-t beamer                                                          \
		--toc                                                              \
		-V classoption:aspectratio=169                                     \
		-V lang=en-US                                                      \
		$<

clean:
	rm *.pdf

.PHONY: clean
