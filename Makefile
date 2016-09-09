problem := parsec
config := config/$(problem).toml
database := output/$(problem).sqlite3
archive := $(problem).tar.gz
artifacts := $(addprefix output/,$(problem).model $(problem).model.meta checkpoint log)

all:

archive: $(archive)

generate: $(database)

clean:
	@rm -rf $(artifacts)

$(archive): $(artifacts)
	@tar -czf $@ $^

$(database): $(config) $(shell find src -name '*.rs')
	@rm -f $@
	@mkdir -p output
	@cargo run -- --verbose --config $<

.PHONY: all archive clean generate
