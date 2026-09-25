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