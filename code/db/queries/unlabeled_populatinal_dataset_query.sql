SELECT
    p.User_input,
    f.*
FROM dim_populational p
INNER JOIN fact_metrics f
    ON p.User_input = f.User_input;