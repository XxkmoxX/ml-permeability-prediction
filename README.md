## Predicting equivalent hydraulic conductivity (_K<sub>eq</sub>_) of synthetic reservoirs using machine learning techniques

This project aims to develop machine learning models for predicting _K<sub>eq</sub>_ based on 2D geometric connectivity indicators. The project includes various models, data preprocessing steps, and evaluation metrics to ensure robust predictions.

### **1 - Introduction**
Reservoir engineering uses numerical simulation to model and predict the behavior of fluids in a reservoir over time. It involves the creation of mathematical and computational models that represent the geological and fluid dynamic properties of the reservoir rock. These models are based on geological data, geophysical surveys, laboratory tests, well tests, production histories, among other sources of information, and allow for the simulation of how fluids move in the Earth's subsurface, as well as how they can be extracted or injected to or from the surface (Steve 2018). This discipline has applications in the energy sector, such as oil and gas extraction, geothermal energy production, groundwater production, underground contaminant transport, and carbon dioxide capture and storage (CCUS).

These models are used to construct maps composed of cells that characterize the reservoir at high-resolution scales. Because simulations at these scales require extensive computational resources, an upscaling procedure is necessary to transfer the petrophysical properties and characteristics of the high-resolution fine grid to a low-resolution simulation grid.
