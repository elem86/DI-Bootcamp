--1.
UPDATE film
SET language_id = 3
WHERE film_id IN (1, 2, 3);

SELECT film_id, title, language_id
FROM film
WHERE film_id IN (1, 2, 3);

--2.
-- customer.store_id references store, and customer.address_id references address. This means new customers can only be inserted using valid store and address IDs that already exist in those tables.

--3.
-- DROP TABLE customer_review;

--4.
SELECT COUNT(*)
FROM rental
WHERE return_date IS NULL;

--5.
SELECT
    film.film_id,
    film.title,
    film.rental_rate
FROM rental
JOIN inventory
    ON rental.inventory_id = inventory.inventory_id
JOIN film
    ON inventory.film_id = film.film_id
WHERE rental.return_date IS NULL
ORDER BY film.rental_rate DESC
LIMIT 30;

--6.
--6.1 
-- SELECT
--     film.description,
--     film.title,
--     actor.first_name,
--     actor.last_name
-- FROM film
-- JOIN film_actor
--     ON film.film_id = film_actor.film_id
-- JOIN actor
--     ON film_actor.actor_id = actor.actor_id
-- WHERE film.description ILIKE '%sumo%'
--   AND actor.first_name = 'Penelope'
--   AND actor.last_name = 'Monroe';

 
--6.2 
SELECT
    length,
    title,
	rating,
	description
FROM film
WHERE length < 60 
	AND rating='R'
	AND description ILIKE '%documentary%';

--6.3
SELECT
    film.title,
    payment.amount,
    rental.return_date
FROM customer
JOIN rental
    ON customer.customer_id = rental.customer_id
JOIN payment
    ON rental.rental_id = payment.rental_id
JOIN inventory
    ON rental.inventory_id = inventory.inventory_id
JOIN film
    ON inventory.film_id = film.film_id
WHERE customer.first_name = 'Matthew'
  AND customer.last_name = 'Mahan'
  AND payment.amount > 4.00
  AND rental.return_date >= '2005-07-28'
  AND rental.return_date < '2005-08-02';

--6.4
SELECT
    film.title,
    film.description,
    film.replacement_cost
FROM customer
JOIN rental
    ON customer.customer_id = rental.customer_id
JOIN inventory
    ON rental.inventory_id = inventory.inventory_id
JOIN film
    ON inventory.film_id = film.film_id
WHERE customer.first_name = 'Matthew'
  AND customer.last_name = 'Mahan'
  AND (
      film.title ILIKE '%boat%'
      OR film.description ILIKE '%boat%'
  )
ORDER BY film.replacement_cost DESC;
