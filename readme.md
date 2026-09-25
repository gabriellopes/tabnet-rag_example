# minimal tabnet rag example

obj -> merge elements from different datasets onto a common pool so newly neural networks architectures can gather data from in order to create agents, perform simulations, and tune the model.

context -> which diseases are more tendly to be present depending on the region and their distance to SUS-programee

## proj structure


. rag, 
    a llm lib will be responsible for retrieving from the vector search, to embedd the received queries, and to generate responses;

        - yaml config for fine-tunement and re-ranking algorithms, developed as plug-ins;

. data,
    pools of data, either in csv or turtle formats, capable of being read through specialized plug-ins for each;


. api,
    the features developed, such as asking against the model, fine-tunements, and re-ranking applying will be served throughout fastAPI endpoints.

 
