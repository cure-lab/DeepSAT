/*
Copyright (c) 2005 - 2007 Armin Biere, Johannes Kepler University.

All rights reserved. Redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  3. All advertising materials mentioning features or use of this software
     must display the following acknowledgement:

	  This product includes software developed by 
	  Armin Biere, Johannes Kepler University, Linz, Austria.

  4. Neither the name of the University nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.
   
THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include "../aiger/aiger.h"

#if 0
#define DUMP_CLAUSES
#define DUMP_REMAINING
#define DUMP_GATES
#define DUMP_ROOTS
#define DUMP_LEAFS
#define DUMP_COVER
#define DUMP_CYCLES
#endif

#if 0
#define LOGMEM(p,b) \
  do { fprintf (stderr, \
                "[cnf2aig] memory %c %d %p\n", \
                ((b) < 0) ? '-' : '+', abs(b), (p)); } while (0)
#else
#define LOGMEM(p,b) do { } while (0)
#endif

#define ADJUST_MAX_BYTES() \
  do { \
    if (current_bytes >= max_bytes) \
      max_bytes = current_bytes; \
  } while (0)

#define NEW(p) \
  do { \
    long bytes = sizeof *(p); \
    (p) = malloc (bytes); \
    memset ((p), 0, bytes); \
    current_bytes += bytes; \
    LOGMEM (p,bytes); \
    ADJUST_MAX_BYTES (); \
  } while (0)

#define DELETE(p) \
  do { \
    long bytes = sizeof *(p); \
    assert (bytes <= current_bytes); \
    current_bytes -= bytes; \
    free (p); \
    LOGMEM (p,-bytes); \
    (p) = 0; \
  } while (0)

#define NEWN(p,n) \
  do { \
    long bytes = (n) * sizeof *(p); \
    (p) = malloc (bytes); \
    memset ((p), 0, bytes); \
    current_bytes += bytes; \
    LOGMEM (p,bytes); \
    ADJUST_MAX_BYTES (); \
  } while (0)

#define DELETEN(p,n) \
  do { \
    long bytes = (n) * sizeof *(p); \
    assert (bytes <= current_bytes); \
    current_bytes -= bytes; \
    free (p); \
    LOGMEM (p,-bytes); \
    (p) = 0; \
  } while (0)

#define REALLOCN(p,o,n) \
  do { \
    long new_bytes = (n) * sizeof *(p); \
    long old_bytes = (o) * sizeof *(p); \
    assert (old_bytes <= current_bytes); \
    current_bytes -= old_bytes; \
    current_bytes += new_bytes; \
    (p) = realloc ((p), new_bytes); \
    LOGMEM (p,-old_bytes); \
    LOGMEM (p,new_bytes); \
    ADJUST_MAX_BYTES (); \
  } while (0)

#define GZIP "gunzip -c %s"

#define NOT(a) ((AIG*)(1 ^ (Word)(a)))
#define SIGN(a) (1 & (Word)(a))
#define STRIP(a) (SIGN(a) ? NOT(a) : (a))

#define TRUE (&true_aig_node)
#define FALSE NOT(TRUE)

typedef unsigned long Word;
typedef struct Clause Clause;
typedef struct ClauseBucket ClauseBucket;
typedef struct TwoLitBucket TwoLitBucket;
typedef struct Variable Variable;
typedef struct Gate Gate;
typedef struct Node Node;
typedef struct AIG AIG;

enum Tag
{
  NIL = 0,
  NOT = 1,
  AND = 2,
  XOR = 3,
  ITE = 4,
};

typedef enum Tag Tag;

struct Gate
{
#ifndef NDEBUG
  Tag tag;
#else
  unsigned tag : 30;
#endif
  unsigned covered : 1;
  unsigned backedge : 1;
  int lhs;
  Node * clauses;
  Gate * next;
};

struct Clause
{
  unsigned size : 30;
  unsigned extracted : 1;
  unsigned covered : 1;
  int * literals;
};

struct ClauseBucket
{
  Clause * clause;
  ClauseBucket * next;
};

struct TwoLitBucket
{
  int lits[2];
  Node * clauses;
  TwoLitBucket * next;
};

struct Variable
{
  unsigned defined : 1;
  unsigned input : 1;
  unsigned leaf : 1;
  unsigned root : 1;
  unsigned used : 1;
  unsigned covered : 1;
  unsigned unit : 1;
  Gate * gate;
  AIG * lhs;
  AIG * rhs;
  int count;
  int mark;
};

struct Node
{
  void * head;
  Node * tail;
};

struct AIG
{
  unsigned idx;		/* non zero iff variable otherwise variable index */
  unsigned id;		/* platform independent hash value */
  unsigned ref;		/* reference counter */
  AIG * c0;		/* left child (child 0) */
  AIG * c1;		/* right child (child 1) */
  AIG * next;		/* collision chain */
  AIG * cache;		/* substitution cache */
  AIG * subst;		/* substitution */
  unsigned tseitin_idx;
  unsigned onstack : 1;
};

static FILE * input;
static int close_input;
static int pclose_input;
static int lineno;
static int max_idx;

static const char * output_name;

static Clause * clauses, * hclauses;
static int szclauses;
static int nclauses;			/* redundant: hclauses - clauses */
static int ccovered;

static Variable * variables;
static int vcovered;
static Node * units;

static Variable ** stack, ** top;
static int szstack;

static ClauseBucket ** table;
static unsigned sztable;

static TwoLitBucket ** twolits;
static unsigned sztwolits;

static int trivial_clauses;
static int duplicated_clauses;
static int xors, nots, backedges, gcovered;

static int * clause, * hclause, * eoc;

static Gate * ugates, * dgates;
static int ngates;

static AIG ** aigs;
static AIG true_aig_node;
static unsigned szaigs, naigs, idaigs;

static size_t current_bytes, max_bytes;

static void
die (const char * fmt, ...)
{
  va_list ap;
  fputs ("*** cnf2aig: ", stderr);
  va_start (ap, fmt);
  vfprintf (stderr, fmt, ap);
  fputc ('\n', stderr);
  fflush (stderr);
  exit (1);
}

static int
next (void)
{
  int res = getc (input);
  if (res == '\n')
    lineno++;
  return res;
}

static int
cmp_lit (const void * p, const void * q)
{
  int a = * (int *) p;
  int b = * (int *) q;
  int c = abs (a);
  int d = abs (b);
  int res = (c - d);

  if (res)
    return res;

  return a - b;
}

static int
equal_clause (Clause * c)
{
  const int * p = c->literals;
  const int * q = clause;

  while (*p && *p++ == *q++)
    ;

  return *p ? 0 : !*q;
}

static unsigned
primes [] =
{
  2000000011,
  2000000033,
  2000000063,
  2000000087,
  2000000089,

  2000000099,
  2000000137,
  2000000141,
  2000000143,
  2000000153,
};

#define NUM_PRIMES  (sizeof (primes) / sizeof (primes[0]))

static unsigned
hash_clause (void)
{
  unsigned res, i, tmp;
  const int  * p;

  i = 0;
  res = 0;

  for (p = clause; (tmp = *p); p++)
    {
      res += tmp * primes [i++];
      if (i >= NUM_PRIMES)
	i = 0;
    }

  return res & (sztable - 1);
}

static ClauseBucket **
find_clause_bucket (void)
{
  ClauseBucket ** p, * b;

  for (p = table + hash_clause ();
       (b = *p) && !equal_clause (b->clause);
       p = &b->next)
    ;

  return p;
}

static Clause *
contains_clause (void)
{
  ClauseBucket * b = *find_clause_bucket ();
  return b ? b->clause : 0;
}

static void
insert_clause (Clause * c)
{
  ClauseBucket ** p, * b;
  p = find_clause_bucket ();
  assert (*p == 0);
  NEW (b);
  b->clause = c;
  *p = b;
}

static unsigned
hash_twolits (int a, int b)
{
  unsigned res;
  assert (abs (a) < abs (b));
  res = primes[0] * (unsigned) a;
  res += (unsigned) b;
  res *= primes[1];
  res &= sztwolits - 1;
  return res;
}

static void
sort_twolits (int * p, int * q)
{
  int a, b;

  a = *p;
  b = *q;

  assert (a != b);
  assert (a != -b);

  if (abs (a) < abs (b))
    return;

  *p = b;
  *q = a;
}

static TwoLitBucket **
find_twolit_bucket (int l0, int l1)
{
  TwoLitBucket ** p, * b;

  assert (abs (l0) < abs (l1));
  for (p = twolits + hash_twolits (l0, l1);
       (b = *p) && (b->lits[0] != l0 || b->lits[1] != l1);
       p = &b->next)
    ;

  return p;
}

static Node *
contains_twolit (int l0, int l1)
{
  TwoLitBucket * b;
  sort_twolits (&l0, &l1);
  b = *find_twolit_bucket (l0, l1);
  return b ? b->clauses : 0;
}

static Node *
cons (void * head, Node * tail)
{
  Node * res;
  NEW (res);
  res->head = head;
  res->tail = tail;
  return res;
}

static void
insert_twolit (Clause * c, int l0, int l1)
{
  TwoLitBucket ** p, * b;
  sort_twolits (&l0, &l1);
  p = find_twolit_bucket (l0, l1);
  b = *p;
  if (!b)
    {
      NEW (b);
      *p = b;
      b->lits[0] = l0;
      b->lits[1] = l1;
    }

  b->clauses = cons (c, b->clauses);
}

static void
sort_clause (int size)
{
  qsort (clause, size, sizeof (clause[0]), cmp_lit);
}

static int
is_redundant (int size)
{
  int i, j, last, this, trivial;

  assert (size >= 0);
  sort_clause (size);

  if (contains_clause ())
    {
      duplicated_clauses++;
      return 1;
    }

  trivial = 0;
  last = 0;
  j = 0;

  for (i = 0; i < size; i++)
    {
      this = clause[i];

      if (last == -this)
	trivial = 1;

      clause[j++] = this;
      last = this;
    }

  if (trivial)
    trivial_clauses++;

  assert (!clause[size]);
  clause[j++] = 0;

  return 0;
}

#if !defined(NDEBUG) || defined(DUMP_REMAINING) || defined(DUMP_CLAUSES)
static void
dump_clause_aux (const char * prefix, Clause * c)
{
  int * p;

  fputs (prefix, stderr);
  for (p = c->literals; *p; p++)
    fprintf (stderr, "%d ", *p);

  fputs ("0\n", stderr);
}
#endif

#if !defined(NDEBUG)
void
dump_clause (Clause * c)
{
  dump_clause_aux ("", c);
}
#endif

#if !defined(NDEBUG) || defined(DUMP_CLAUSES)
void
dump_clauses (void)
{
  Clause * p;

  fflush (stderr);
  for (p = clauses; p < hclauses; p++)
    dump_clause_aux ("c CLAUSE ", p);
  fflush (stderr);
}
#endif

static void
parse (void)
{
  int ch, n, dn, lit, sign, size;
  Clause * c;

  lineno = 1;

SKIP_COMMENTS:
  ch = next ();
  if (isspace (ch))
    goto SKIP_COMMENTS;

  if (ch == 'c')
    {
      while ((ch = next ()) != '\n' && ch != EOF)
	;

      goto SKIP_COMMENTS;
    }

  if (ch != 'p')
    die ("expected 'c' or 'p' at line %d", lineno);

  if (fscanf (input, " cnf %d %d", &max_idx, &szclauses) != 2 ||
      szclauses < 0 || max_idx < 0)
    die ("invalid header at line %d", lineno);

  fprintf (stderr, "[cnf2aig] found 'p cnf %d %d'\n", max_idx, szclauses);

  NEWN (clauses, szclauses);
  hclauses = clauses;

  for (sztable = 1; sztable <= szclauses; sztable *= 2)
    ;
  NEWN (table, sztable);

  sztwolits = 2 * sztable;
  NEWN (twolits, sztwolits);

  NEWN (variables, max_idx + 1);

  dn = n = 0;
  for (;;)
    {
      ch = next ();
      if (isspace (ch))
	continue;

      if (ch == EOF)
	{
	  if (hclause > clause)
	    die ("trailing '0' missing at EOF");

	  if (n < szclauses)
	    die ("%d clauses missing at EOF", szclauses - n);

	  fprintf (stderr, "[cnf2aig] parsed %d clauses\n", szclauses);

	  if (trivial_clauses)
	    fprintf (stderr,
		     "[cnf2aig] found %d trivial clauses\n",
		     trivial_clauses);

	  if (duplicated_clauses)
	    fprintf (stderr,
		     "[cnf2aig] found %d duplicated clauses\n",
		     duplicated_clauses);

	  nclauses = hclauses - clauses;
	  if (trivial_clauses || duplicated_clauses)
	    fprintf (stderr, "[cnf2aig] kept %d clauses\n", nclauses);

	  fflush (stderr);
#ifdef DUMP_CLAUSES
	  dump_clauses ();
#endif
	  return;
	}

      if (ch == '-')
	{
	  sign = -1;
	  ch = next ();
	}
      else
	sign = 1;

      if (!isdigit (ch))
	die ("expected digit but got 0x%02x", ch);

      lit = ch - '0';
      while (isdigit (ch = next ()))
	lit = 10 * lit + (ch - '0');

      if (lit > max_idx)
	die ("maximum variable index exceeded in '%d' at line %d",
	     sign * lit, lineno);

      if (n == szclauses)
	die ("additional clause at line %d", lineno);

      lit *= sign;

      if (hclause == eoc)
	{
	  unsigned old_count = hclause - clause;
	  unsigned new_count = old_count ? 2 * old_count : 1;
	  REALLOCN (clause, old_count, new_count);
	  eoc = clause + new_count;
	  hclause = clause + old_count;
	}

      *hclause++ = lit;

      if (lit)
	continue;

      n++;
      size = hclause - clause - 1;
      hclause = clause;

      if (is_redundant (size))
	continue;

      c = hclauses++;
      c->size = size;
      NEWN (c->literals, size + 1);
      memcpy (c->literals, clause, (size + 1) * sizeof (c->literals[0]));
      insert_clause (c);

      if (size != 3)
	continue;

      insert_twolit (c, c->literals[0], c->literals[1]);
      insert_twolit (c, c->literals[0], c->literals[2]);
      insert_twolit (c, c->literals[1], c->literals[2]);
    }
}

static Node *
cdr (Node * node)
{
  assert (node);
  return node->tail;
}

static void *
car (Node * node)
{
  assert (node);
  return node->head;
}

static void
free_nodes (Node * node)
{
  Node * p, * next;

  for (p = node; p; p = next)
    {
      next = cdr (p);
      DELETE (p);
    }
}

static void
release_clauses (void)
{
  Clause * p;

  for (p = clauses; p < hclauses; p++)
    DELETEN (p->literals, p->size + 1);

  DELETEN (clauses, szclauses);
}

static void
release_table (void)
{
  ClauseBucket * p, * next;
  int i;

  for (i = 0; i < sztable; i++)
    for (p = table[i]; p; p = next)
      {
	next = p->next;
	DELETE (p);
      }

  DELETEN (table, sztable);
}

static void
release_twolits (void)
{
  TwoLitBucket * p, * next;
  int i;
  for (i = 0; i < sztwolits; i++)
    for (p = twolits[i]; p; p = next)
      {
	next = p->next;
	free_nodes (p->clauses);
	DELETE (p);
      }

  DELETEN (twolits, sztwolits);
}

static void
release_variables (void)
{
  DELETEN (variables, max_idx + 1);
}

static void
release_gates (void)
{
  Gate * p, * next;
  int i;

  for (i = 0; i <= 1; i++)
    for (p = i ? dgates : ugates; p; p = next)
      {
	next = p->next;
	free_nodes (p->clauses);
	DELETE (p);
      }
}

static void
add_clause_to_gate (Clause * c, Gate * g)
{
  g->clauses = cons (c, g->clauses);
  c->extracted = 1;
}

static void
add_gate_to_lhs (Gate * gate, int lhs)
{
  Variable  * v;

  v = variables + abs(lhs);
  if (v->gate)
    {
#ifdef DUMP_GATES
      fprintf (stderr, "c MULTIPLE GATES FOR VARIABLE %d\n", v - variables);
#endif
    }
  else
    v->gate = gate;
}

#if !defined(NDEBUG) || defined(DUMP_GATES)
void
dump_gate (Gate * g)
{
  int i, lit, printed, sign;
  Clause * c;
  Node * p;

  fprintf (stderr, "c GATE ");
  switch (g->tag)
    {
      case AND:
	fprintf (stderr, "AND %d = ", g->lhs);
	for (p = g->clauses; p; p = cdr (p))
	  {
	    c = car (p);
	    if (c->size == 2)
	      continue;

	    printed = 0;
	    for (i = 0; i < c->size; i++)
	      {
		lit = c->literals[i];
		if (lit == g->lhs)
		  continue;

		if (printed++)
		  fputs (" & ", stderr);

		fprintf (stderr, "%d", -lit);
	      }
	  }
	break;
      case ITE:
	fprintf (stderr, "ITE %d = ...", g->lhs);
	break;
      case XOR:
	fprintf (stderr, "XOR ...");
	break;
      default:
	assert (g->tag == NOT);
	fprintf (stderr, "NOT ");
	assert (g->clauses);
	c = car (g->clauses);
	sign = (c->literals[0] < 0) ? -1 : 1;
	assert (c->size == 2);
	fprintf (stderr,
	         "%d = %d",
		 sign * c->literals[0], -sign * c->literals[1]);
	break;
    }
  fputc ('\n', stderr);
}

void
dump_gates (void)
{
  Gate * g;
  int i;
  fflush (stderr);
  for (i = 0; i <= 1; i++)
    for (g = i ? dgates : ugates; g; g = g->next)
      dump_gate (g);
  fflush (stderr);
}
#endif

static Gate *
new_gate (int lhs, Tag tag)
{
  Gate * res;
  NEW (res);
  res->tag = tag;
  res->lhs = lhs;

  if (tag == XOR || tag == NOT)
    {
      res->next = ugates;
      ugates = res;
    }
  else
    {
      res->next = dgates;
      dgates = res;
    }

  ngates++;
  return res;
}

static void
new_unit (Clause * c)
{
  Variable * v;
  assert (c->size == 1);
  v = variables + abs (c->literals[0]);
  v->unit = 1;
  units = cons (c, units);
}

static void
extract_units (void)
{
  int units = 0;
  Clause * p;

  for (p = clauses; p < hclauses; p++)
    {
      if (p->size != 1)
	continue;

      new_unit (p);
      units++;
    }

  if (units)
    {
      fprintf (stderr, "[cnf2aig] found %d units\n", units);
      fflush (stderr);
    }
}

static Gate *
not_gate (void)
{
  return new_gate (0, NOT);
}

static void
extract_nots (void)
{
  Clause * p, * c;
  Gate * g;

  nots = 0;
  for (p = clauses; p < hclauses; p++)
    {
      if (p->size != 2)
	continue;

      if (p->literals[0] < 0)
	continue;

      clause[0] = -p->literals[0];
      clause[1] = -p->literals[1];
      clause[2] = 0;

      c = contains_clause ();
      if (!c)
	continue;

      g = not_gate ();

      add_clause_to_gate (p, g);
      add_clause_to_gate (c, g);

      nots++;
    }

  if (nots)
    {
      fprintf (stderr, "[cnf2aig] found %d NOT gates\n", nots);
      fflush (stderr);
    }
}

static Gate *
and_gate (int lhs)
{
  Gate * res = new_gate (lhs, AND);
  add_gate_to_lhs (res, lhs);
  return res;
}

static void
extract_ands (void)
{
  int i,  j, len, ands, found, lhs;
  Clause * p, * c;
  Gate * gate;

  len = ands = 0;

  for (p = clauses; p < hclauses; p++)
    {
      if (p->size <= 2)
	continue;

      clause[2] = 0;

      for (i = 0; p->literals[i]; i++)
	{
	  found = 1;
	  for (j = 0; found && p->literals[j]; j++)
	    {
	      if (i == j)
		continue;

	      clause[i > j] = -p->literals[i];
	      clause[i < j] = -p->literals[j];
	      if (!contains_clause ())
		found = 0;
	    }

	  if (!found)
	    continue;

	  lhs = p->literals[i];
	  gate = and_gate (lhs);

	  add_clause_to_gate (p, gate);

	  len += p->size;
	  ands++;

	  for (j = 0; found && p->literals[j]; j++)
	    {
	      if (i == j)
		continue;

	      clause[i > j] = -p->literals[i];
	      clause[i < j] = -p->literals[j];
	      c = contains_clause ();
	      assert (c);
	      add_clause_to_gate (c, gate);
	    }
	}
    }

  if (ands)
    {
      fprintf (stderr, 
	       "[cnf2aig] found %d AND gates (arity %.1f)\n", 
	       ands, (len - ands) / (double) ands);
      fflush (stderr);
    }
}

static unsigned
parity (unsigned n)
{
  unsigned res = 0;

  while (n)
    {
      if (n&1)
	res = !res;

      n >>= 1;
    }

  return res;
}

static void
signed_copy_of_clause (Clause * c, unsigned signs)
{
  unsigned i;

  memcpy (clause, c->literals, (c->size + 1) * sizeof clause[0]);

  for (i = 0; signs; i++)
    {
      if (signs & 1)
	clause[i] *= -1;

      signs >>= 1;
    }
}

static Gate *
xor_gate (void)
{
  return new_gate (0, XOR);
}

static void
extract_xors (void)
{
  int len, i, found, count;
  Clause * p, * c;
  unsigned signs;
  Gate * g;

  len = xors = 0;

  for (p = clauses; p < hclauses; p++)
    {
      if (p->size < 3)
	continue;

      if (p->size >= 30)		/* one billon clauses required */
	break;

      for (i = 0; i < p->size; i++)
	if (p->literals[i] > 0)
	  break;			/* forces continue below ... */

      if (i <= p->size)
	continue;

      found = 1;
      count = (1 << p->size);
      for (signs = 0; found && signs < count; signs++)
	{
	  if (parity (signs))
	    continue;

	  signed_copy_of_clause (p, signs);
	  found = (contains_clause () != 0);
	}

      if (!found)
	continue;

      g = xor_gate ();
      for (signs = 0; signs < count; signs++)
	{
	  if (parity (signs))
	    continue;

	  signed_copy_of_clause (p, signs);
	  c = contains_clause ();
	  assert (c);
	  add_clause_to_gate (c, g);
	}

      len += p->size;
      xors++;
    }

  if (xors)
    {
      fprintf (stderr,
	       "[cnf2aig] found %d XOR gates (arity %.1f)\n",
	       xors, (len - xors) / (double) xors);
      fflush (stderr);
    }
}

static Clause *
find_ternary_clause (int a, int b, int c)
{
  assert (abs(a) != abs (b));
  assert (abs(b) != abs (c));
  assert (abs(a) != abs (c));
  clause[0] = a;
  clause[1] = b;
  clause[2] = c;
  clause[3] = 0;
  sort_clause (3);
  return contains_clause ();
}

static Gate *
ite_gate (int lhs)
{
  Gate * res = new_gate (lhs, ITE);
  add_gate_to_lhs (res, lhs);
  return res;
}

static void
extract_ites (void)
{
  int ites, i, j, k, lhs, cond, then_lit, else_lit, tmp;
  Clause * tr, * tl, *el, *er;
  Node * p;
  Gate * g;

  ites = 0;

  for (tr = clauses; tr < hclauses; tr++)
    {
      if (tr->size != 3)
	continue;

      for (i = 0; i < 3; i++)
	{
	  for (j = 0; j < 3; j++)
	    {
	      if (i == j)
		continue;

	      lhs = tr->literals[i];
	      if (lhs > 0)
		continue;

	      lhs = -lhs;

	      cond = tr->literals[j];
	      if (cond > 0)
		continue;

	      cond = -cond;

	      then_lit = 0;
	      for (k = 0; k < 3; k++)
		{
		  tmp = tr->literals[k];

		  if (tmp == -lhs)
		    continue;

		  if (tmp == -cond)
		    continue;

		  then_lit = tmp;
		  break;
		}

	      assert (then_lit);

	      tl = find_ternary_clause (lhs, -cond, -then_lit);
	      if (!tl)
		continue;

	      for (p = contains_twolit (-lhs, cond); p; p = cdr (p))
		{
		  er = car (p);
		  else_lit = er->literals[0];
		  if (else_lit == -lhs || else_lit == cond)
		    {
		      else_lit = er->literals[1];
		      if (else_lit == -lhs || else_lit == cond)
			else_lit = er->literals[2];
		    }

		  if (else_lit == -then_lit)
		    continue;			/* actually xor */

		  assert (else_lit != -lhs);
		  assert (else_lit != cond);

		  el = find_ternary_clause (lhs, cond, -else_lit);
		  if (!el)
		    continue;

		  ites++;

		  g = ite_gate (lhs);
		  add_clause_to_gate (tr, g);
		  add_clause_to_gate (tl, g);
		  add_clause_to_gate (er, g);
		  add_clause_to_gate (el, g);
		}
	    }
	}
    }

  if (ites)
    {
      fprintf (stderr, "[cnf2aig] found %d ITE gates\n", ites);
      fflush (stderr);
    }
}

static void
reverse_gates (Gate ** start)
{
  Gate * this = 0, * next = * start;

  while (next)
    {
      Gate * tmp = next->next;
      next->next = this;
      this = next;
      next = tmp;
    }

  *start = this;
}

static void
extract (void)
{
  int extracted, remaining, count;
  Clause * p;

  count = hclauses - clauses;
  if (!count)
    return;

  extract_units ();
  extract_nots ();
  extract_ands ();
  extract_xors ();
  extract_ites ();

  remaining = 0;
  for (p = clauses; p < hclauses; p++)
    {
      if (p->extracted)
	continue;

      if (p->size <= 1)
	continue;

#ifdef DUMP_REMAINING
      dump_clause_aux ("c UNEXTRACTED CLAUSE ", p);
#endif
      remaining++;
    }
  extracted = count - remaining;

  reverse_gates (&ugates);
  reverse_gates (&dgates);

  fprintf (stderr,
	   "[cnf2aig] extracted %d gates from %d clauses (%.1f%%)\n",
	   ngates, extracted, 100.0 * extracted / (double) count);

  if (remaining)
    fprintf (stderr,
	     "[cnf2aig] %d remaining unextracted clauses (%.1f%%)\n",
	     remaining, 100.0 * remaining / (double) count);

#ifdef DUMP_GATES
  dump_gates ();
#endif
}

static int
break_cycles (void)
{
  int i, cyclic, lit, idx, rootidx, rootsfirst, res;
  Variable * v, * u, * root;
  const int * q;
  Clause * c;
  Node * p;

  res = 0;
  top = stack;

  for (rootsfirst = 1; rootsfirst >= 0; rootsfirst--)
    {
      for (rootidx = 1; rootidx <= max_idx; rootidx++)
	{
	  root = variables + rootidx;

	  if (!root->defined)
	    continue;

	  if (root->mark)
	    continue;

	  if (rootsfirst && !root->root)
	    continue;

	  *top++ = root;
	  root->mark = 1;

	  while (top > stack)
	    {
	      v = *--top;

	      if (v)
		{
		  assert (v->defined);

		  if (v->mark == 3)		/* already hit */
		    continue;

		  assert (v->mark == 1);
		  assert (v->gate);
		  
		  cyclic = 0;
		  for (p = v->gate->clauses; !cyclic && p; p = cdr (p))
		    {
		      c = car (p);
		      for (q = c->literals; !cyclic && (lit = *q); q++)
			{
			  idx = abs (lit);
			  u = variables + idx;

			  if (u == v)
			    continue;

			  if (u->mark == 3)	/* cross edge */
			    continue;

			  if (u->mark == 2)	/* back edge */
			    cyclic = 1;
			}
		    }

		  if (cyclic)
		    {
		      v->defined = 0;
		      res++;
#ifdef DUMP_CYCLES
		      fprintf (stderr,
			       "c CYCLE %u -> %u\n", idx, v - variables);
#endif
		    }
		  else
		    {
		      *top++ = v;
		      *top++ = 0;
		      v->mark = 2;		/* mark as 'on stack' */

		      for (p = v->gate->clauses; p; p = cdr (p))
			{
			  c = car (p);
			  for (q = c->literals; (lit = *q); q++)
			    {
			      idx = abs (lit);
			      u = variables + idx;
			      if (u == v)
				continue;

			      if (u->mark)
				continue;

			      if (!u->defined)
				continue;

			      *top++ = u;
			      u->mark = 1;
			    }
			}
		    }
		}
	      else
		{
		  assert (top > stack);
		  v = *--top;
		  assert (v->mark == 2);
		  v->mark = 3;
		  assert (v->defined);			/* keep it */
		}
	    } /* end of 'while (top > stack)' */
	} /* end of 'for (rootidx = 1; ...' */
    } /* end of 'for (rootsfirst = 1; ...' */

  /* Reset mark field of all variables.
   */
  for (i = 1; i <= max_idx; i++)
    variables[i].mark = 0;

  return res;
}

/* TODO factor this in digestable portions ... */

static void
direct (void)
{
  int ndirectedxors, ndirectednots, ndirectedinputs, ndirectedroots, broken;
  int nroots, nleafs, ndefined, ninputs, nfloating, ndirected, nunits;
  int i, lit, lhs, last, round, nzombies, dcount;
  Variable * v, * u;
  Clause * c;
  Node * p;
  Gate * g;

  if (!max_idx)
    return;

  nzombies = max_idx;
  for (c = clauses; c < hclauses; c++)
    {
      for (i = 0; i < c->size; i++)
	{
	  lit = abs (c->literals[i]);
	  v = variables + lit;
	  if (!v->used)
	    {
	      assert (nzombies > 0);
	      nzombies--;
	      v->used = 1;
	    }
	}
    }

  if (nzombies)
    fprintf (stderr,
	     "[cnf2aig] %d unused zombie variables (%.1f%%)\n", 
	     nzombies, 100.0 * nzombies / (double) max_idx);

  ndefined = 0;
  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;

      if (!v->gate)
	continue;

      v->defined = 1;		/* TODO: probably need to break cycles here */
      ndefined++;
    }

  if (ndefined)
    fprintf (stderr,
	     "[cnf2aig] forced %d gate outputs (%.1f%%)\n", 
	     ndefined, 100.0 * ndefined / (double) max_idx);

  nroots = ndefined;
  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;
      if (v->defined)
	v->root = 1;
    }

  ninputs = 0;
  for (g = dgates; g; g = g->next)
    {
      lhs = abs (g->lhs);
      assert (lhs);
      assert (variables[lhs].defined);

      for (p = g->clauses; p; p = cdr (p))
	{
	  c = car (p);
	  for (i = 0; i < c->size; i++)
	    {
	      lit = abs (c->literals[i]);

	      if (lit == lhs)
		continue;

	      v = variables + lit;
	      if (!v->input)
		{
		  ninputs++;
		  v->input = 1;
		}

	      if (v->root)
	        {
		  assert (nroots > 0);
		  v->root = 0;
		  nroots--;
		}
	    }
	}
    }
 
  for (g = ugates; g; g = g->next)
    for (p = g->clauses; p; p = cdr (p))
      {
	c = car (p);
	for (i = 0; i < c->size; i++)
	  {
	    lit = abs (c->literals[i]);
	    v = variables + lit;
	    if (!v->root)
	      continue;

	    assert (nroots > 0);
	    v->root = 0;
	    nroots--;
	  }
      }

  if (ninputs)
    fprintf (stderr,
	     "[cnf2aig] forced %d gate inputs (%.1f%%)\n", 
	     ninputs, 100.0 * ninputs / (double) max_idx);

  if (nroots)
    fprintf (stderr,
	     "[cnf2aig] forced %d roots (%.1f%%)\n", 
	     nroots, 100.0 * nroots / (double) max_idx);

#ifdef DUMP_ROOTS
  for (i = 1; i <= max_idx; i++)
    if (variables[i].root)
      fprintf (stderr, "c FORCED ROOT %d\n", i);
#endif

  if ((broken = break_cycles ()))
    {
      fprintf (stderr,
	       "[cnf2aig] removed %d cyclic definitions\n",
	       broken);
      assert (ndefined >= broken);
      ndefined -= broken;
    }

  nleafs = 0;
  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;
      if (v->input && !v->defined)
	{
	  v->leaf = 1;
	  nleafs++;
	}
    }

  for (g = ugates; g; g = g->next)
    {
      assert (g->clauses);
      for (p = g->clauses; p; p = cdr (p))
	{
	  c = car (p);
	  for (i = 0; i < c->size; i++)
	    {
	      lit = abs (c->literals[i]);

	      v = variables + lit;

	      if (v->leaf)
		{
		  assert (v->input);
		  assert (!v->defined);
		  assert (nleafs > 0);

		  v->leaf = 0;
		  nleafs--;
		}
	    }
	}
    }

  if (nleafs)
    fprintf (stderr,
	     "[cnf2aig] forced %d leafs (%.1f%%)\n", 
	     nleafs, 100.0 * nleafs / (double) max_idx);

#ifdef DUMP_LEAFS
  for (i = 1; i <= max_idx; i++)
    if (variables[i].leaf)
      fprintf (stderr, "c FORCED LEAF %d\n", i);
#endif
  /* TODO: we should use a working list and direct all gates possible.  The
   * work list has to be a priority queue or at least consist of two work
   * lists, one of of those gates that have less than 2 undirected inputs.
   * If exactly one undirected input exists, and all the other inputs are
   * defined then this input becomes the output and the gate is directed.
   * Otherwise a conflict occurred and this gate will never be covered.
   * It will be kept in the ugates list.  The other working list consists
   * of gates with more than just one undetermined input.  Additionally we
   * may want to split each of the working lists in XOR and NOT lists.
   */
  ndirectedinputs = ndirected = ndirectednots = ndirectedxors = 0;
  for (round = 0; round < 10000; round++)
    {
      last = ndirected;
      for (g = ugates; g; g = g->next)
	{
	  if (g->lhs)
	    continue;

	  assert (g->clauses);
	  c = car(g->clauses);

	  dcount = 0;
	  for (i = 0; i < c->size; i++)
	    {
	      lit = abs (c->literals[i]);
	      v = variables + lit;

	      if (v->defined)
		dcount++;
	    }

	  if (dcount == c->size)
	    continue;

	  v = 0;
	  lhs = 0;

	  if (dcount + 1 == c->size)
	    {
	      for (i = 0; i < c->size; i++)
		{
		  lhs = abs (c->literals[i]);
		  v = variables + lhs;
		  if (!v->defined)
		    break;
		}
	    }
	  else
	    continue;

	  assert (lhs);
	  for (i = 0; i < c->size; i++)
	    {
	      if (c->literals[i] > 0)
		lhs = -lhs;

	      lit = abs (c->literals[i]);
	      if (lit == lhs)
		continue;

	      u = variables + lit;
	      if (u->input)
		continue;

	      u->input = 1;
	      ninputs++;
	      ndirectedinputs++;
	    }

	  assert (v);
	  assert (!v->defined);
	  v->defined = 1;

	  g->lhs = lhs;
	  add_gate_to_lhs (g, lhs);

	  ndefined++;
	  ndirected++;

	  if (g->tag == XOR)
	    ndirectedxors++;

	  if (g->tag == NOT)
	    ndirectednots++;
	}

      if (last == ndirected)
	break;

      fprintf (stderr,
	       "[cnf2aig] directed %d gates in round %d\n",
	       ndirected - last, round);
    }

  if (ndirected)
    fprintf (stderr,
	     "[cnf2aig] directed %d gates and %d inputs in %d rounds\n",
	     ndirected, ndirectedinputs, round);

  if (ndirectedxors)
    fprintf (stderr,
	     "[cnf2aig] directed %d XOR gates (%.1f%%)\n",
	     ndirectedxors,
	     100.0 * ndirectedxors / (double) xors);

  if (ndirectednots)
    fprintf (stderr,
	     "[cnf2aig] directed %d NOT gates (%.1f%%)\n",
	     ndirectednots,
	     100.0 * ndirectednots / (double) nots);

  ndirectedroots = 0;
  nroots = 0;

  for (g = ugates; g; g = g->next)
    {
      lhs = abs (g->lhs);
      if (!lhs)
	continue;
      v = variables + lhs;
      assert (!v->root);

      if (!v->defined)		/* may be cyclic */
	continue;

      v->root = 1;
      nroots++;
      ndirectedroots++;
    }

  for (g = ugates; g; g = g->next)
    {
      lhs = abs (g->lhs);
      for (p = g->clauses; p; p = cdr (p))
	{
	  c = car (p);
	  for (i = 0; i < c->size; i++)
	    {
	      lit = abs (c->literals[i]);
	      if (lit == lhs)
		continue;

	      v = variables + lit;
	      if (!v->root)
		continue;

	      assert (v->defined);
	      assert (v->gate);
	      assert (v->gate->tag == XOR || v->gate->tag == NOT);

	      v->root = 0;

	      assert (ndirectedroots > 0);
	      ndirectedroots--;

	      assert (nroots > 0);
	      nroots--;
	    }
	}
    }

  if (ndirectedroots)
    fprintf (stderr,
	     "[cnf2aig] directed %d roots (%.1f%%)\n",
	     ndirectedroots, 100.0 * ndirectedroots / (double) max_idx);

#ifdef DUMP_ROOTS
  for (g = ugates; g; g = g->next)
    {
      lhs = abs (g->lhs);
      if (!lhs)
	continue;

      v = variables + lhs;
      if (!v->root)
	continue;

      assert (v->defined);
      assert (v->gate);

      if (v->gate->tag != XOR && v->gate->tag != NOT)
	continue;

      fprintf (stderr, "c DIRECTED ROOT %d\n", lhs);
    }
#endif

  assert (nroots <= ndefined);

  if ((broken = break_cycles ()))
    {
      fprintf (stderr,
	       "[cnf2aig] removed %d cyclic definitions\n",
	       broken);
      assert (ndefined >= broken);
      ndefined -= broken;
    }

  nunits = 0;
  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;

      if (v->root)
	continue;

      if (v->leaf)
	continue;

      if (v->defined)
	continue;

      if (!v->unit)
	continue;

      nunits++;
    }

  if (nunits)
    fprintf (stderr, "[cnf2aig] %d units definitions\n", nunits);

  nfloating = max_idx - nleafs - ndefined - nzombies - nunits;
  assert (nfloating >= 0);

  if (nfloating)
    fprintf (stderr,
	     "[cnf2aig] %d remaining floating variables (%.1f%%)\n", 
	     nfloating, 100.0 * nfloating / (double) max_idx);
}

static void
vpush (Variable * v)
{
  assert (top < stack + szstack);
  assert (!v->mark);
  v->mark = 1;
  assert (v->mark);
  *top++ = v;
}

static void 
adjust_count_clause (int except, Clause * c, int delta)
{
  const int * p;
  int lit, idx;
  Variable * v;

  assert (except > 0);
  assert (delta);

  for (p = c->literals; (lit = *p); p++)
    {
      idx = abs (lit);
      if (idx == except)
	continue;

      v = variables + idx;

#ifndef NDEBUG
      if (delta > 0)
	assert (v->count + delta > v->count);
      else
	assert (v->count >= -delta);
#endif

      v->count += delta;
      if (delta < 0 && !v->count)
	vpush (v);
    }
}

static void
adjust_count_gate (Variable * v, Gate * gate, int delta)
{
  Node * p;

  assert (gate);
  assert (gate->lhs);
  assert (abs (gate->lhs) == v - variables);

  for (p = gate->clauses; p; p = cdr (p))
    adjust_count_clause (v - variables, car (p), delta);
}

static void
cover_clauses_of_gate (Gate * gate)
{
  Clause * c;
  Node * p;

  assert (gate);
  assert (gate->lhs);

  for (p = gate->clauses; p; p = cdr (p))
    {
      c = car (p);
      if (c->covered)
	continue;

      c->covered = 1;
      ccovered++;
#ifdef DUMP_COVER
      dump_clause_aux ("c COVERED CLAUSE ", c);
#endif
    }
}

static void
cover (void)
{
  Variable * v;
  Clause * c;
  Node * p;
  int i;

  top = stack;

  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;
      if (!v->gate)
	continue;

      adjust_count_gate (v, v->gate, 1);
    }

  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;

      if (!v->used)
	continue;

      if (v->count)
	continue;

      if (!v->gate)
	continue;

      /* This 'v' is not necessarily a root, since it may have become
       * a root after breaking a cycle.
       */
      vpush (v);
    }

  while (top > stack)		/* TODO: no cyclic definitions left? */
    {
      v = *--top;
      v->covered = 1;
      vcovered++;
#ifdef DUMP_COVER
      fprintf (stderr,
	       "c COVERING VARIABLE %d THROUGH DEFINITION\n",
	       v - variables);
#endif
      assert (!v->count);
      if (!v->gate)
	continue;

      gcovered++;
      v->gate->covered = 1;
      adjust_count_gate (v, v->gate, -1);
    }

  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;

      if (v->covered)
	continue;

      if (!v->unit)
	continue;

      v->covered++;
      vcovered++;
#ifdef DUMP_COVER
      fprintf (stderr,
	       "c COVERING VARIABLE %d THROUGH UNIT CLAUSE\n",
	       v - variables);
#endif
    }

  if (max_idx)
    {
      if (max_idx == vcovered)
	fprintf (stderr,
	         "[cnf2aig] covered all %d variables (100.0%%)\n",
		 max_idx);
      else
	fprintf (stderr,
		 "[cnf2aig] covered %d out of %d variables (%.1f%%)\n",
		 vcovered, max_idx, 100.0 * vcovered / (double) max_idx);
    }

  if (ngates)
    {
      if (gcovered == ngates)
	fprintf (stderr,
	         "[cnf2aig] covered all %d gates (100.0%%)\n",
		 ngates);
      else
	fprintf (stderr,
		 "[cnf2aig] covered %d out of %d gates (%.1f%%)\n",
		 gcovered, ngates, 100.0 * gcovered / (double) ngates);
    }

  if (backedges)
    fprintf (stderr,
	     "[cnf2aig] cut %d cyclic gates (%.1f%%)\n",
	     backedges, 100.0 * backedges / (double) ngates);

  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;
      if (!v->covered)
	continue;

      if (v->gate && v->gate->covered)
	cover_clauses_of_gate (v->gate);
    }

  for (p = units; p; p = cdr (p))
    {
      c = car (p);
      assert (!c->covered);

      v = variables + abs (c->literals[0]);

      if (!v->leaf)
	continue;

      c->covered = 1;
      ccovered++;
    }

  if (nclauses)
    {
      if (ccovered == nclauses)
	  fprintf (stderr,
		   "[cnf2aig] covered all %d clauses (100.0%%)\n",
		   nclauses);
      else
	fprintf (stderr,
		 "[cnf2aig] covered %d out of %d clauses (%.1f%%)\n",
		 ccovered, nclauses, 100.0 * ccovered / (double) nclauses);
    }
}

static void
release_aigs (void)
{
  AIG * p, * next;
  int i;

  for (i = 0; i < szaigs; i++)
    {
      for (p = aigs[i]; p; p = next)
	{
	  next = p->next;
	  DELETE (p);
	}
    }

  DELETEN (aigs, szaigs);
}

static unsigned
hash_aig (unsigned idx, AIG * c0, AIG * c1)
{
  unsigned res;

  assert (NUM_PRIMES >= 3);

  res = idx;
  res *= primes[0];

  if (c0)
    {
      assert (c0 != FALSE);
      assert (c0 != TRUE);

      if (SIGN (c0))
	{
	  res += 1;
	  c0 = NOT (c0);
	}

      res += 2 * c0->id;
      res *= primes[1];

      assert (c1);
      assert (c1 != FALSE);
      assert (c1 != TRUE);

      if (SIGN (c1))
	{
	  res += 1;
	  c1 = NOT (c1);
	}

      res += 2 * c1->id;
      res *= primes[2];

      assert (c0->id <= c1->id);
    }
  else
    assert (!c1);

  res &= szaigs - 1;

  return res;
}

static void
enlarge_aigs (void)
{
  AIG ** old_aigs = aigs, * p, * next;
  unsigned old_szaigs = szaigs, i, h;

  szaigs *= 2;
  NEWN (aigs, szaigs);

  for (i = 0; i < old_szaigs; i++)
    for (p = old_aigs[i]; p; p = next)
      {
	next = p->next;
	h = hash_aig (p->idx, p->c0, p->c1);
	p->next = aigs[h];
	aigs[h] = p;
      }

  DELETEN (old_aigs, old_szaigs);
}

static AIG **
find_aig (unsigned idx, AIG * c0, AIG * c1)
{
  unsigned h = hash_aig (idx, c0, c1);
  AIG ** p, * a;

  for (p = aigs + h;
       (a = *p) && (a->idx != idx || a->c0 != c0 || a->c1 != c1);
       p = &a->next)
    ;

  return p;
}

static AIG *
inc_aig (AIG * original)
{
  AIG * stripped = STRIP (original);
  if (stripped != TRUE)
    {
      assert (stripped->ref);
      stripped->ref++;
    }
  else
    assert (stripped != FALSE);

  return original;
}

static void
dec_aig (AIG * aig)
{
  AIG ** p;

  return;				/* remove and fix wrong dec_aig */

  if (SIGN (aig))
    aig = NOT (aig);

  if (aig == TRUE)
    return;

  assert (aig != FALSE);

  assert (aig->ref);
  aig->ref--;
  if (aig->ref)
    return;

  if (!aig->idx)
    {
      dec_aig (aig->c0);		/* derecursivy */
      dec_aig (aig->c1);
    }

  p = find_aig (aig->idx, aig->c0, aig->c1);
  assert (*p == aig);
  *p = aig->next;
  assert (naigs);
  naigs--;
  DELETE (aig);
}

static AIG *
new_aig (unsigned idx, AIG * c0, AIG * c1)
{
  AIG * res;
  NEW (res);
  res->idx = idx;
  res->c0 = c0;
  res->c1 = c1;
  res->id = ++idaigs;		/* true_aig_node.id == 0 */
  res->cache = 0;
  res->subst = 0;
  res->tseitin_idx = 0;
  res->next = 0;
  res->ref = 1;			/* give back a reference ! */
  return res;
}

static AIG *
and_aig (AIG * a, AIG * b)
{
  AIG ** p, * res;

  if (a == FALSE || b == FALSE || a == NOT (a))
    return FALSE;

  if (a == TRUE || a == b)
    return inc_aig (b);

  if (b == TRUE)
    return inc_aig (a);

  if (STRIP (a)->id > STRIP (b)->id)
    {
      AIG * tmp = a;
      a = b;
      b = tmp;
    }

  if (naigs >= szaigs)
    enlarge_aigs ();

  p = find_aig (0, a, b);
  if (!*p)
    {
      res = new_aig (0, a, b);
      inc_aig (a);
      inc_aig (b);
      naigs++;
      *p = res;
    }
  else
    res = inc_aig (*p);

  return res;
}

static AIG *
lit2aig (int lit)
{
  unsigned idx = abs (lit);
  AIG * res, ** p;

  assert (lit);

  p = find_aig (idx, 0, 0);
  if (!*p)
    {
      res = new_aig (idx, 0, 0);
      naigs++;
      *p = res;
    }
  else
    res = inc_aig (*p);

  if (lit < 0)
    res = NOT (res);

  return res;
}

static AIG *
or_aig (AIG * a, AIG * b)
{
  return NOT (and_aig (NOT (a), NOT (b)));
}

static AIG * 
clause2aig (Clause * c, int lhs)
{
  AIG * res, * tmp, * a;
  const int * p;
  int lit;

  res = FALSE;
  for (p = c->literals; (lit = *p); p++)
    {
      if (lit == lhs)
	a = TRUE;
      else if (lit == -lhs)
	a = FALSE;
      else
	a = lit2aig (lit);

      tmp = or_aig (res, a);
      dec_aig (a);
      dec_aig (res);
      res = tmp;
    }

  return res;
}

static AIG *
gate2aig (Gate * g)
{
  AIG * res, * tmp, * a;
  Clause * c;
  Node * p;

  res = TRUE;
  for (p = g->clauses; p; p = cdr (p))
    {
      c = car (p);
      a = clause2aig (c, g->lhs);
      tmp = and_aig (res, a);
      dec_aig (a);
      dec_aig (res);
      res = tmp;
    }

  return res;
}

/* TODO derecursivy this function */

static AIG *
substitute (AIG * aig)
{
  AIG * res, * l, * r;
  int sign = SIGN (aig);

  if (sign)
    aig = STRIP (aig);

  if (aig == TRUE)
    res = aig;
  else
    {
      assert (aig != FALSE);
      res = aig->cache;
      if (!res)
	{
	  if (aig->idx)
	    {
	      if (aig->subst)
		{
		  assert (!aig->onstack);
		  aig->onstack = 1;
		  res = substitute (aig->subst);
		  assert (aig->onstack);
		  aig->onstack = 0;
		}
	      else
		res = inc_aig (aig);
	    }
	  else
	    {
	      l = substitute (aig->c0);
	      r = substitute (aig->c1);
	      res = and_aig (l, r);
	      assert (!aig->cache);
	      dec_aig (r);
	      dec_aig (l);
	    }

	  aig->cache = res;
	}

      inc_aig (res);
    }
  
  if (sign)
    res = NOT (res);

  return res;
}

static unsigned
aig2idx (AIG * aig)
{
  int sign = SIGN (aig);
  unsigned res;

  if (aig == TRUE)
    return 1;

  if (aig == FALSE)
    return 0;

  if (sign)
    aig = NOT (aig);

  res = aig->tseitin_idx;
  assert (res);

  if (sign)
    res++;

  return res;
}

static unsigned
tseitin (aiger * mgr, AIG * aig, unsigned idx)
{
  int sign = SIGN (aig);
  if (sign)
    aig = NOT (aig);

  if (aig == TRUE)
    return idx;

  assert (aig != FALSE);

  if (aig->tseitin_idx)
    return idx;

  if (!aig->idx)
    {
      idx = tseitin (mgr, aig->c0, idx);
      idx = tseitin (mgr, aig->c1, idx);
    }

  aig->tseitin_idx = idx;
  idx += 2;

  if (aig->idx)
    {
      static char buffer[20];
      sprintf (buffer, "%d", aig->idx);
      aiger_add_input (mgr, aig2idx (aig), buffer);
    }
  else
    aiger_add_and (mgr, aig2idx (aig), aig2idx (aig->c0), aig2idx (aig->c1));

  return idx;
}

static int
print (AIG * aig)
{
  aiger * mgr;
  int res;

  mgr = aiger_init ();

  tseitin (mgr, aig, 2);
  aiger_add_output (mgr, aig2idx (aig), "SATISFY");
  aiger_add_comment (mgr, "cnf2aig");

  if (output_name)
    aiger_add_comment (mgr, output_name);
  if (output_name)
    res = aiger_open_and_write_to_file (mgr, output_name);
  else
    res = aiger_write_to_file (mgr, aiger_ascii_mode, stdout);

  aiger_reset (mgr);

  return res;
}

static AIG *
build (void)
{
  AIG * tmp, * res, * new;
  unsigned count = 0;
  int i, lit, idx;
  Variable * v;
  Clause *c;
  Node * p;

  szaigs = (1 << 10);
  NEWN (aigs, szaigs);

  count = 0;
  for (i = 1; i <= max_idx; i++)
    {
      v = variables + i;
      if (!v->gate || !v->gate->covered)
	continue;

      assert (v->gate->lhs == i || v->gate->lhs == -i);

      v->lhs = lit2aig (i);
      v->rhs = gate2aig (v->gate);

      assert (abs (v->gate->lhs) == i);
      if (v->gate->lhs < 0)
	v->rhs = NOT (v->rhs);

      assert (!SIGN (v->lhs));
      v->lhs->subst = inc_aig (v->rhs);

      count++;
    }

  if (count)
    fprintf (stderr,
	     "[cnf2aig] definitions need %d aig nodes (%.1f on average)\n",
	     count, naigs / (double) count);

  count = 0;
  for (p = units; p; p = cdr (p))
    {
      c = car (p);
      lit = c->literals[0];
      idx = abs (lit);
      v = variables + idx;

      if (!v->leaf)
	continue;

      if (v->lhs)
	continue;		/* two unit clauses with this variable  */

      count++;

      v->lhs = lit2aig (idx);
      v->rhs = (lit < 0) ? FALSE : TRUE;

      assert (!SIGN (v->lhs));
      v->lhs->subst = inc_aig (v->rhs);
    }

  if (count)
    fprintf (stderr, "[cnf2aig] %d leaf units\n", count);

  count = 0;
  res = TRUE;
  for (c = clauses; c < hclauses; c++)
    {
      if (c->covered)
	continue;
#ifdef DUMP_REMAINING
      dump_clause_aux ("c UNCOVERED CLAUSE ", c);
#endif
      tmp = clause2aig (c, 0);
      new = substitute (tmp);
      dec_aig (tmp);
      tmp = and_aig (res, new);
      dec_aig (new);
      dec_aig (res);
      res = tmp;

      count++;
    }

  fprintf (stderr, "[cnf2aig] substituted %d uncovered clauses\n", count);

  return res;
}

int
main (int argc, char ** argv)
{
  AIG * aig;

  /* TODO: make option parsing more robust */

  if (argc > 3)
    die ("usage: cnf2aig [<dimacs-file>[.gz] [<aig-file>]]");

  if (argc >= 2)
    {
      const char * suffix;
      suffix = strstr (argv[1], ".gz");

      if (suffix && (argv[1] + strlen (argv[1]) - 3 == suffix))
	{
	  char * cmd;
	  int len = strlen (argv[1]) + strlen (GZIP);
	  NEWN (cmd, len);
	  sprintf (cmd, GZIP, argv[1]);
	  input = popen (cmd, "r");
	  DELETEN (cmd, len);
	  if (!input)
	    die ("can not read gzipped filed '%s'", argv[1]);

	  pclose_input = 1;
	}
      else
	{
	  input = fopen (argv[1], "r");
	  if (!input)
	    die ("can not read '%s'", argv[1]);

	  close_input = 1;
	}

      fprintf (stderr, "[cnf2aig] reading '%s'\n", argv[1]);
      fflush (stderr);
    }

  if (argc >= 3)
    output_name = argv[2];

  if (!input)
    input = stdin;

  parse ();
  extract ();

  release_table ();
  release_twolits ();

  szstack = 2 * max_idx;
  NEWN (stack, szstack);

  direct ();
  cover ();
  aig = build ();
  if (!print (aig))
    die ("writing to '%s' failed", output_name ? output_name : "<stdout>");

  release_gates ();
  release_clauses ();
  release_variables ();

/* TODO make the following assertion work */
#if 0
#ifndef NDEBUG
  dec_aig (aig);
  assert (!naigs);
#endif
#endif
  release_aigs ();

  free_nodes (units);

  DELETEN (clause, eoc - clause);
  DELETEN (stack, szstack);

  fprintf (stderr, "[cnf2aig] allocated %.1f MB\n",
	   max_bytes / (double)(1<<20));

  if (current_bytes)
    fprintf (stderr, "[cnf2aig] leaked %.0f bytes\n", (double) current_bytes);

  if (close_input)
    fclose (input);

  if (pclose_input)
    pclose (input);

  return 0;
}
