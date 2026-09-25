# minimal tabnet rag example

obj -> merge elements from different datasets onto a common pool so newly neural networks architectures can gather data from in order to create agents, perform simulations, and tune the model.

context -> which diseases are more tendly to be present depending on the region and their distance to SUS-programee

    . Adjusted Mortality Rate=f(Disease Incidence,PBF Coverage,ESF Primary Care Coverage,Income Per Capita)

## proj structure


. rag, 
    a llm lib will be responsible for retrieving from the vector search, to embedd the received queries, and to generate responses;

        - yaml config for fine-tunement and re-ranking algorithms, developed as plug-ins;

. data,
    pools of data, either in csv or turtle formats, capable of being read through specialized plug-ins for each;


. api,
    the features developed, such as asking against the model, fine-tunements, and re-ranking applying will be served throughout fastAPI endpoints.

 

### query examples

A. Racial Disparity in Early vs. Late DiagnosisQuestion: 

    "Are racial minorities experiencing higher relative incidence of occupational cancer in municipalities with low health compliance rates?"
    
    Analytical Approach: 
        Cross-tabulate SINAN's CS_RACA against IBGE's population racial/demographic baseline to compute Racial Disparity Indices (RDI) per municipality, then correlate RDI against PBF's perc_acomp_saude.


B. The Protective Buffer of Welfare ConditionalityQuestion: 

    "Does high Bolsa Família prenatal compliance (perc_gestantes_prenatal_em_dia) correlate with lower disease rates among vulnerable women in primary care sectors?"
    
    Analytical Approach: Evaluate the interaction between MDS compliance metrics and SINAN disease notifications (ID_AGRAVO), controlling for total municipal population (pop) and primary care infrastructure (CNES facility count).


C. Primary Care Access DeficitQuestion: 

    "Which municipalities show high disease incidence despite adequate primary care infrastructure (CNES density), indicating barriers in healthcare utilization?"
    
    Analytical Approach: Model Disease Rate vs. Facility Density ($CNES\_count / pop$), using SPARQL in our Knowledge Graph to isolate structural health bottlenecks.

    