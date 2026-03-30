## Improve search algorithm
The current search algorithm relies on a semi-coherent STFT search statistic to dentify individual track candidates. Based on the best fitting track, a SVD basis is constructed and single EMRI tracks are constructed with JAX and fitted using a hybrid of differential evolution nd the Adam optimizer. This is used to find the best fitting track. Returning to the physical parameter space is complicated. 

### Task 1
Investigate the structure of the repository and identify key components related to the search algorithm. Summarize in a markdown file called search_algorithm.md. 
- Identify the main steps in the search algorithm
- Point me to the key functions and files implementing these steps
- Outline the high-level flow of the algorithm.  

### Task 2
Identify potential avenues for improvement in the search algorithm. Use the information in @../JAX-waveform/gradient_identification.md to inform your analysis. Add your findings to a new section in the search_algorithm.md markdown file. Pay special attention to the fact that I now have fast, vmappable and differentiable EMRI tracks in JAX, in a sparse basis. This should allow for more efficient optimization and potentially a new approach to returning to the physical parameter space. 

### Task 3
Based on your analysis in Task 2, propose a new approach to accelerate the search algorithm. In particular, consider techniques that start from a single identified track in the STFT basis. Using the plunge time estimated from the STFT track, suggest methods to efficiently recover more tracks. Also consider using a particle swarm optimization approach to quickly scan the paremeter space based off information inferred from a single f, fdot track. Add the two proposals to the search_algorithm.md file. 

### Task 4
Implement a prototype of the proposed approach in task 3. This should include:
- code structure for the new approach
- key functions and their implementations
- integration points with the existing search algorithm
- Extension of the search_algorithm.md file to include the new approach and its implementation details.

## Context
EMRI detection is a computationally expensive task, for the following reasons: 
- the parameter space is large, and the posterior is very narrow. This implies a massive prior to posterior ratio, rendering a brute-force matched full template bank search infeasible. 
- the waveform generation is expensive O(100)ms for a full waveform
-  The parameter space is highly correlated and this leads to multi-modal posteriors. While the dominant mode is exponentially larger than the others, the other modes are still significant and can slow down the search for stochastic samplers when they get stuck in a local maximum.

## Technical details
Leverage the power of JAX for efficient optimization, gradient descent, batching and automatic differentiation. Use the existing codebase as a foundation, and ensure that the new implementation is compatible with the current structure of the repository. Focus on optimizing the search algorithm by utilizing the differentiable EMRI tracks in JAX, and explore techniques to efficiently recover more tracks based on the plunge time estimated from the STFT track. 

## Deliverables
- A markdown file (search_algorithm.md) summarizing the current search algorithm and potential improvements.
- A prototype implementation of the proposed approach to accelerate the search algorithm, integrated with the existing codebase.
- Updated documentation in the search_algorithm.md file detailing the new approach and its implementation.
- An example notebook in the folder Examples/ demonstrating the new search algorithm in action, comparing its performance to the existing method.
- A set of files for profiling the new implementation on a GPU. 