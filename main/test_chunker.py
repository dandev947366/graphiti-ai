# test_chunker.py

from .chunker import chunk_text


def test_chunking():
    sample_text = (
        "A black hole is a massive, compact astronomical object so dense that its gravity prevents anything from escaping, even light. "
        "Albert Einstein's theory of general relativity predicts that a sufficiently compact mass will form a black hole.[3] "
        "The boundary of no escape is called the event horizon. A black hole has a great effect on the fate and circumstances of an object crossing it, "
        "but has no locally detectable features according to general relativity.[4] In many ways, a black hole acts like an ideal black body, as it reflects no light.[5][6] "
        "Quantum field theory in curved spacetime predicts that event horizons emit Hawking radiation, with the same spectrum as a black body of a temperature inversely proportional to its mass. "
        "This temperature is of the order of billionths of a kelvin for stellar black holes, making it essentially impossible to observe directly. "
        "Objects whose gravitational fields are too strong for light to escape were first considered in the 18th century by John Michell and Pierre-Simon Laplace. "
        "In 1916, Karl Schwarzschild found the first modern solution of general relativity that would characterise a black hole. "
        "Due to his influential research, the Schwarzschild metric is named after him. David Finkelstein, in 1958, first published the interpretation of 'black hole' as a region of space from which nothing can escape. "
        "Black holes were long considered a mathematical curiosity; it was not until the 1960s that theoretical work showed they were a generic prediction of general relativity. "
        "The discovery of neutron stars by Jocelyn Bell Burnell in 1967 sparked interest in gravitationally collapsed compact objects as a possible astrophysical reality. "
        "The first black hole known was Cygnus X-1, identified by several researchers independently in 1971.[7][8] "
        "Black holes typically form when massive stars collapse at the end of their life cycle. After a black hole has formed, it can grow by absorbing mass from its surroundings. "
        "Supermassive black holes of millions of solar masses may form by absorbing other stars and merging with other black holes, or via direct collapse of gas clouds. "
        "There is consensus that supermassive black holes exist in the centres of most galaxies."
    )

    chunks = chunk_text(sample_text, max_words=20)

    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk.split())} words) ---")
        print(chunk)


if __name__ == "__main__":
    test_chunking()
