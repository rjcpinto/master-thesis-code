SELECT
    d.User_input,
    f.*
FROM dim_pharmacogenomics d
LEFT JOIN fact_metrics f
    ON d.User_input = f.User_input;