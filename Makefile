config := config/parsec.toml
database := $(shell \grep '^path' $(config) | \head -1 | \cut -d'"' -f2)
database_sources := $(shell find src -name '*.rs')
name := $(basename $(notdir $(database)))
archive := $(name).tar.gz
archive_sources := $(addprefix $(dir $(database)),$(name).model $(name).model.meta checkpoint log)

all: database

archive: $(archive)

database: $(database)

$(archive): $(archive_sources)
	tar -czf $@ $^

$(database): $(config) $(database_sources)
	@rm -f $@
	@mkdir -p `dirname $@`
	@cargo run -- --verbose --config $<

.PHONY: all archive database
