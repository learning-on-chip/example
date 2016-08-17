days := 1
config := config/parsec.toml
sources := $(shell find src -name '*.rs')

all: tests/fixtures/database.sqlite3

%.sqlite3: $(sources)
	@rm -f $@
	@mkdir -p `dirname $@`
	@cargo run -- --verbose --config $(config) --length $(($(days) * 60 * 60)) --output $@

.PHONY: all
