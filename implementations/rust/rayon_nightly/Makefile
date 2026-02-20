.PHONY: build clean

CARGO = $(HOME)/.cargo/bin/cargo
MANIFEST = implementations/rust/rayon_nightly/Cargo.toml

build:
	$(CARGO) +nightly build --release --manifest-path $(MANIFEST)

clean:
	$(CARGO) clean --manifest-path $(MANIFEST)
