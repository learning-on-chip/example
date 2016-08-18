config := config/parsec.toml
sources := $(shell find src -name '*.rs')

tests/fixtures/database.sqlite3: $(config) $(sources)
	@rm -f $@
	@mkdir -p `dirname $@`
	@cargo run -- --verbose --config $<

.PHONY: all
