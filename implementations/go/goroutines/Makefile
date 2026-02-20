.PHONY: build clean

# Try user-local Go install first, then system PATH
GO ?= $(shell which go 2>/dev/null || echo $(HOME)/.local/go-sdk/go/bin/go)

OUTDIR = build
TARGET = $(OUTDIR)/libportfolio_go.so
SRC    = portfolio_compute.go

$(TARGET): $(SRC)
	mkdir -p $(OUTDIR)
	$(GO) build -buildmode=c-shared -o $(TARGET) $(SRC)

build: $(TARGET)

clean:
	rm -rf $(OUTDIR)
