# FunctionStudio
💡 Announcing the Development of a new Optimization Library

As an AI professional, I'm excited to share that I'm developing a new optimization library, planned for release later this year. This library aims to address key challenges in the field:

❓ Why another optimization library?
- Current tools like Scikit-learn, PyTorch, and TensorFlow offer limited support for multi-objective optimization. This library simplifies finding trade-off models for any combination of objectives, aimed to ease the use of multi-objective optimization in model development.
- Users will have the ability to choose whether to train models end-to-end or as separate modules. This feature is particularly useful for optimizing decisions composed of multiple sub-decisions.
- Built on a low-code language (DMN) for both the input and output layers, the library ensures interpretability, interoperability, and interchangeability of model chains.
- The library will support single-objective reinforcement learning, multi-objective reinforcement learning, supervised learning, and multi-objective supervised learning, all through a single, user-friendly interface.


To do
- Implement class called Function allowing to fit any multi-objective function by adding objectives.✅ 
- Implement supervised example case on how to use function class ✅
- Implement Reinforcement learning example for function ✅ 
- Implement CompositeFunction class ✅
- Implement reinforcement learning example for CompositeFunction class ✅
- Implement multi-objective reinforcement learning example ✅
- Implement the reporting module to visualize the found pareto front❌ 
- Implement the dmn input and output layer to convert function from and to dmn❌ 
- Setup pypi package, requirements file❌ 
- Make unit tests❌ 
- document classes in code❌ 
