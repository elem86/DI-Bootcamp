--  Exercise 1: Movie Rankings and Analysis
--1.

SELECT
    g.genre_name,
    m.title,
    RANK() OVER (
        PARTITION BY g.genre_name
        ORDER BY m.popularity DESC
    ) AS popularity_rank
FROM movies.movie m
JOIN movies.movie_genres mg
    ON m.movie_id = mg.movie_id
JOIN movies.genre g
    ON mg.genre_id = g.genre_id
ORDER BY g.genre_name, popularity_rank;

--2.
SELECT
    pc.company_name,
    m.title,
    m.revenue,
    NTILE(4) OVER (
        PARTITION BY pc.company_name
        ORDER BY m.revenue DESC
    ) AS revenue_quartile
FROM movies.movie m
JOIN movies.movie_company mc
    ON m.movie_id = mc.movie_id
JOIN movies.production_company pc
    ON mc.company_id = pc.company_id
ORDER BY pc.company_name, revenue_quartile, m.revenue DESC;



--3.
SELECT
    g.genre_name,
    m.title,
    m.budget,
    SUM(m.budget) OVER (
        PARTITION BY g.genre_name
        ORDER BY m.release_date, m.movie_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total_budget
FROM movies.movie m
JOIN movies.movie_genres mg
    ON m.movie_id = mg.movie_id
JOIN movies.genre g
    ON mg.genre_id = g.genre_id
ORDER BY g.genre_name, m.release_date;


--4.
SELECT DISTINCT
    g.genre_name,
    FIRST_VALUE(m.title) OVER (
        PARTITION BY g.genre_name
        ORDER BY m.release_date DESC
    ) AS most_recent_movie,
    FIRST_VALUE(m.release_date) OVER (
        PARTITION BY g.genre_name
        ORDER BY m.release_date DESC
    ) AS release_date
FROM movies.movie m
JOIN movies.movie_genres mg
    ON m.movie_id = mg.movie_id
JOIN movies.genre g
    ON mg.genre_id = g.genre_id
ORDER BY g.genre_name;

--  Exercise 2: Cast and Crew Performance Analysis

--1.

SELECT
    actor_name,
    movie_count,
    DENSE_RANK() OVER (
        ORDER BY movie_count DESC
    ) AS actor_rank
FROM (
    SELECT
        p.person_name AS actor_name,
        COUNT(DISTINCT mc.movie_id) AS movie_count
    FROM movies.person p
    JOIN movies.movie_cast mc
        ON p.person_id = mc.person_id
    GROUP BY p.person_id, p.person_name
) AS actor_counts
ORDER BY actor_rank, actor_name;


--2.
WITH director_ratings AS (
    SELECT
        p.person_name AS director_name,
        AVG(m.vote_average) AS average_rating
    FROM movies.person p
    JOIN movies.movie_crew mc
        ON p.person_id = mc.person_id
    JOIN movies.movie m
        ON mc.movie_id = m.movie_id
    WHERE mc.job = 'Director'
    GROUP BY p.person_id, p.person_name
),
ranked_directors AS (
    SELECT
        director_name,
        average_rating,
        RANK() OVER (
            ORDER BY average_rating DESC
        ) AS director_rank
    FROM director_ratings
)
SELECT
    director_name,
    ROUND(average_rating, 2)
FROM ranked_directors
WHERE director_rank = 1;


--3.
SELECT
    p.person_name AS actor_name,
    m.title,
    m.release_date,
    m.revenue,
    SUM(m.revenue) OVER (
        PARTITION BY p.person_id
        ORDER BY m.release_date, m.movie_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
FROM movies.person p
JOIN movies.movie_cast mc
    ON p.person_id = mc.person_id
JOIN movies.movie m
    ON mc.movie_id = m.movie_id
ORDER BY p.person_name, m.release_date;


--4.
WITH director_budgets AS (
    SELECT
        p.person_name AS director_name,
        SUM(m.budget) AS total_budget
    FROM movies.person p
    JOIN movies.movie_crew mc
        ON p.person_id = mc.person_id
    JOIN movies.movie m
        ON mc.movie_id = m.movie_id
    WHERE mc.job = 'Director'
    GROUP BY p.person_id, p.person_name
),
ranked_directors AS (
    SELECT
        director_name,
        total_budget,
        RANK() OVER (
            ORDER BY total_budget DESC
        ) AS budget_rank
    FROM director_budgets
)
SELECT
    director_name,
    total_budget
FROM ranked_directors
WHERE budget_rank = 1;

