config := config/parsec.toml
output := $(shell \grep '^path' $(config) | \head -1 | \cut -d'"' -f2)
sources := $(shell find src -name '*.rs')

$(output): $(config) $(sources)
	@rm -f $@
	@mkdir -p `dirname $@`
	@cargo run -- --verbose --config $<

.PHONY: all
