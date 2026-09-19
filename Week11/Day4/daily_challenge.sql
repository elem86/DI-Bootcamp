--1.

WITH budget_growth AS (
    SELECT
        pc.company_id,
        pc.company_name,
        m.title,
        m.release_date,
        m.budget,
        LAG(m.budget) OVER (
            PARTITION BY pc.company_id
            ORDER BY m.release_date, m.movie_id
        ) AS previous_budget
    FROM movies.movie m
    JOIN movies.movie_company mc
        ON m.movie_id = mc.movie_id
    JOIN movies.production_company pc
        ON mc.company_id = pc.company_id
),
growth_rates AS (
    SELECT
        company_id,
        company_name,
        CASE
            WHEN previous_budget IS NULL OR previous_budget = 0 THEN NULL
            ELSE
                ((budget - previous_budget) * 100.0 / previous_budget)
        END AS growth_rate
    FROM budget_growth
)
SELECT
    company_name,
    AVG(growth_rate) AS average_budget_growth_rate
FROM growth_rates
GROUP BY company_id, company_name
ORDER BY average_budget_growth_rate DESC;

--2.

WITH movie_ratings AS (
    SELECT
        movie_id,
        vote_average,
        AVG(vote_average) OVER () AS overall_average
    FROM movies.movie
    WHERE vote_average IS NOT NULL
),
actor_counts AS (
    SELECT
        p.person_id,
        p.person_name,
        COUNT(DISTINCT mr.movie_id) AS high_rated_movies
    FROM movie_ratings mr
    JOIN movies.movie_cast mc
        ON mr.movie_id = mc.movie_id
    JOIN movies.person p
        ON mc.person_id = p.person_id
    WHERE mr.vote_average > mr.overall_average
    GROUP BY p.person_id, p.person_name
),
ranked_actors AS (
    SELECT
        person_name,
        high_rated_movies,
        RANK() OVER (
            ORDER BY high_rated_movies DESC
        ) AS actor_rank
    FROM actor_counts
)
SELECT
    person_name,
    high_rated_movies
FROM ranked_actors
WHERE actor_rank = 1;

--3.
SELECT
    g.genre_name,
    m.title,
    m.release_date,
    m.revenue,
    AVG(m.revenue) OVER (
        PARTITION BY g.genre_id
        ORDER BY m.release_date, m.movie_id
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_average_revenue
FROM movies.movie m
JOIN movies.movie_genres mg
    ON m.movie_id = mg.movie_id
JOIN movies.genre g
    ON mg.genre_id = g.genre_id
ORDER BY g.genre_name, m.release_date;

--4.

WITH keyword_movies AS (
    SELECT
        k.keyword_id,
        k.keyword_name,
        m.movie_id,
        m.title,
        m.revenue,
        COUNT(*) OVER (
            PARTITION BY k.keyword_id
        ) AS movies_in_series,
        SUM(m.revenue) OVER (
            PARTITION BY k.keyword_id
        ) AS total_series_revenue
    FROM movies.movie m
    JOIN movies.movie_keywords mk
        ON m.movie_id = mk.movie_id
    JOIN movies.keyword k
        ON mk.keyword_id = k.keyword_id
),
series_totals AS (
    SELECT DISTINCT
        keyword_id,
        keyword_name,
        movies_in_series,
        total_series_revenue
    FROM keyword_movies
    WHERE movies_in_series > 1
),
ranked_series AS (
    SELECT
        keyword_name,
        movies_in_series,
        total_series_revenue,
        RANK() OVER (
            ORDER BY total_series_revenue DESC
        ) AS revenue_rank
    FROM series_totals
)
SELECT
    keyword_name AS movie_series,
    movies_in_series,
    total_series_revenue
FROM ranked_series
WHERE revenue_rank = 1;