.PHONY: build clean

JAVAC   = javac
JAR     = jar
SRC     = src/com/portfoliosimulator/PortfolioCompute.java
CLASSES = dist/classes
JARFILE = dist/portfolio_vector_api.jar

$(JARFILE): $(SRC)
	mkdir -p $(CLASSES)
	$(JAVAC) --add-modules jdk.incubator.vector \
	         -d $(CLASSES) \
	         $(SRC)
	$(JAR) cf $(JARFILE) -C $(CLASSES) .

build: $(JARFILE)

clean:
	rm -rf dist/classes $(JARFILE)
