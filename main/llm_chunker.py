import re
import requests
from typing import List, Generator, Optional
from dataclasses import dataclass
import json


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:11434/api/generate"
    temperature: float = 0.5
    model: str = "mistral"


@dataclass
class ChunkingConfig:
    max_tokens: int = 512
    min_tokens: int = 50
    stride: int = 50
    clean_text: bool = True


class MistralChunker:
    def __init__(
        self, llm_config: LLMConfig, chunk_config: Optional[ChunkingConfig] = None
    ):
        self.llm_config = llm_config
        self.config = chunk_config if chunk_config else ChunkingConfig()

    def _call_mistral(self, prompt: str) -> str:
        """Call the local Mistral model through Ollama API"""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.llm_config.model,
            "prompt": prompt,
            "temperature": self.llm_config.temperature,
            "stream": False,
        }

        try:
            response = requests.post(
                self.llm_config.base_url, headers=headers, data=json.dumps(data)
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call Mistral model: {e}")

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning pipeline"""
        # Remove citations and references
        text = re.sub(r"\[[\d,]+\]|\([^)]*?\d{4}[^)]*?\)", "", text)
        # Normalize whitespace and clean special chars
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_token_count(self, text: str) -> int:
        """Estimate token count using Mistral"""
        prompt = f"Estimate token count for this text (number only): {text}"
        response = self._call_mistral(prompt)
        try:
            return int(re.search(r"\d+", response).group())
        except (AttributeError, ValueError):
            return len(text.split())  # Fallback

    def semantic_split(self, text: str) -> List[str]:
        """Use Mistral to find semantically meaningful split points"""
        prompt = f"""
        You are a document chunking assistant. Your goal is to split the given technical documentation into semantically meaningful, LLM-friendly chunks.

**Requirements:**
- Break the document into coherent chunks, each containing approximately 300–500 tokens.
- Avoid cutting off in the middle of paragraphs or bullet points.
- Preserve formatting such as lists, code blocks, markdown titles, and image captions where useful.
**Input Document:**

        
        Text: {text}
        """

        response = self._call_mistral(prompt)
        return [s.strip() for s in response.split("|||") if s.strip()]

    def chunk_text(self, text: str) -> Generator[str, None, None]:
        """Generate semantically coherent chunks"""
        if self.config.clean_text:
            text = self.clean_text(text)

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        current_chunk = []
        current_token_count = 0

        for para in paragraphs:
            para_tokens = self.get_token_count(para)

            # Handle oversized paragraphs
            if para_tokens > self.config.max_tokens:
                if current_chunk:
                    yield " ".join(current_chunk)
                    current_chunk = []
                    current_token_count = 0

                # Split large paragraph semantically
                segments = self.semantic_split(para)
                for seg in segments:
                    seg_tokens = self.get_token_count(seg)
                    if current_token_count + seg_tokens <= self.config.max_tokens:
                        current_chunk.append(seg)
                        current_token_count += seg_tokens
                    else:
                        if current_chunk:
                            yield " ".join(current_chunk)
                        current_chunk = [seg]
                        current_token_count = seg_tokens
                continue

            # Normal chunk building
            if current_token_count + para_tokens <= self.config.max_tokens:
                current_chunk.append(para)
                current_token_count += para_tokens
            else:
                if current_chunk:
                    yield " ".join(current_chunk)
                current_chunk = [para]
                current_token_count = para_tokens

        # Final chunk
        if current_chunk and current_token_count >= self.config.min_tokens:
            yield " ".join(current_chunk)

    def chunk_document(self, text: str) -> List[dict]:
        """Process document into chunks with metadata"""
        return [
            {
                "text": chunk,
                "chunk_id": i,
                "token_count": self.get_token_count(chunk),
                "model": self.llm_config.model,
            }
            for i, chunk in enumerate(self.chunk_text(text))
        ]


# Example Usage
if __name__ == "__main__":
    # Configure with your local endpoint
    llm_config = LLMConfig(
        base_url="http://127.0.0.1:11434/api/generate", temperature=0.5, model="mistral"
    )

    chunk_config = ChunkingConfig(max_tokens=384, stride=64, clean_text=True)

    chunker = MistralChunker(llm_config, chunk_config)

    document = """
    From Wikipedia, the free encyclopedia
For other uses, see Black hole (disambiguation).
Blackness of space with black marked as centre of donut of orange and red gases
Composite image of the core region of Messier 87 taken at radio wavelengths showing glowing gas surrounding a supermassive black hole.[1]

Simulated view of a Schwarzschild black hole in front of the Large Magellanic Cloud. Note the gravitational lensing effect, which produces two enlarged but highly distorted views of the Cloud. Across the top, the Milky Way disk appears distorted into an arc.[2]
A black hole is a massive, compact astronomical object so dense that its gravity prevents anything from escaping, even light. Albert Einstein's theory of general relativity predicts that a sufficiently compact mass will form a black hole.[3] The boundary of no escape is called the event horizon. A black hole has a great effect on the fate and circumstances of an object crossing it, but has no locally detectable features according to general relativity.[4] In many ways, a black hole acts like an ideal black body, as it reflects no light.[5][6] Quantum field theory in curved spacetime predicts that event horizons emit Hawking radiation, with the same spectrum as a black body of a temperature inversely proportional to its mass. This temperature is of the order of billionths of a kelvin for stellar black holes, making it essentially impossible to observe directly.

Objects whose gravitational fields are too strong for light to escape were first considered in the 18th century by John Michell and Pierre-Simon Laplace. In 1916, Karl Schwarzschild found the first modern solution of general relativity that would characterise a black hole. Due to his influential research, the Schwarzschild metric is named after him. David Finkelstein, in 1958, first published the interpretation of "black hole" as a region of space from which nothing can escape. Black holes were long considered a mathematical curiosity; it was not until the 1960s that theoretical work showed they were a generic prediction of general relativity. The discovery of neutron stars by Jocelyn Bell Burnell in 1967 sparked interest in gravitationally collapsed compact objects as a possible astrophysical reality. The first black hole known was Cygnus X-1, identified by several researchers independently in 1971.[7][8]

Black holes typically form when massive stars collapse at the end of their life cycle. After a black hole has formed, it can grow by absorbing mass from its surroundings. Supermassive black holes of millions of solar masses may form by absorbing other stars and merging with other black holes, or via direct collapse of gas clouds. There is consensus that supermassive black holes exist in the centres of most galaxies.

The presence of a black hole can be inferred through its interaction with other matter and with electromagnetic radiation such as visible light. Matter falling toward a black hole can form an accretion disk of infalling plasma, heated by friction and emitting light. In extreme cases, this creates a quasar, some of the brightest objects in the universe. Stars passing too close to a supermassive black hole can be shredded into streamers that shine very brightly before being "swallowed."[9] If other stars are orbiting a black hole, their orbits can be used to determine the black hole's mass and location. Such observations can be used to exclude possible alternatives such as neutron stars. In this way, astronomers have identified numerous stellar black hole candidates in binary systems and established that the radio source known as Sagittarius A*, at the core of the Milky Way galaxy, contains a supermassive black hole of about 4.3 million solar masses.

History
The idea of a body so big that even light could not escape was briefly proposed by English astronomical pioneer and clergyman John Michell and independently by French scientist Pierre-Simon Laplace. Both scholars proposed very large stars rather than the modern model of stars with extraordinary density.[10]

Michell's idea, in a short part of a letter published in 1784, calculated that a star with the same density but 500 times the radius of the sun would not let any emitted light escape; the surface escape velocity would exceed the speed of light. Michell correctly noted that such supermassive but non-radiating bodies might be detectable through their gravitational effects on nearby visible bodies.[10][11][12]

In 1796, Laplace mentioned that a star could be invisible if it were sufficiently large while speculating on the origin of the Solar System in his book Exposition du Système du Monde. Franz Xaver von Zach asked Laplace for a mathematical analysis, which Laplace provided and published in journal edited by von Zach.[10]

Scholars of the time were initially excited by the proposal that giant but invisible 'dark stars' might be hiding in plain view, but enthusiasm dampened when the wavelike nature of light became apparent in the early nineteenth century,[13] since light was understood as a wave rather than a particle, it was unclear what, if any, influence gravity would have on escaping light waves.[10][12]

General relativity
See also: History of general relativity
General relativity
Spacetime curvature schematic
G
μ
ν
+
Λ
g
μ
ν
=
κ
T
μ
ν
{\displaystyle G_{\mu \nu }+\Lambda g_{\mu \nu }={\kappa }T_{\mu \nu }}
Introduction
HistoryTimelineTests
Mathematical formulation
Fundamental concepts
Phenomena
Kepler problemGravitational lensingGravitational redshiftGravitational time dilationGravitational wavesFrame-draggingGeodetic effectEvent horizonSingularityBlack hole
Spacetime
Spacetime diagramsMinkowski spacetimeEinstein–Rosen bridge
EquationsFormalisms
Solutions
Scientists
icon Physics portal Category
vte
In 1915, Albert Einstein developed his theory of general relativity, having earlier shown that gravity does influence light's motion. Only a few months later, Karl Schwarzschild found a solution to the Einstein field equations that describes the gravitational field of a point mass and a spherical mass.[14][15] A few months after Schwarzschild, Johannes Droste, a student of Hendrik Lorentz, independently gave the same solution for the point mass and wrote more extensively about its properties.[16][17] This solution had a peculiar behaviour at what is now called the Schwarzschild radius, where it became singular, meaning that some of the terms in the Einstein equations became infinite. The nature of this surface was not quite understood at the time.
    """

    print("Generated Chunks:")
    for chunk in chunker.chunk_document(document):
        print(chunk["text"])
        print()
