Original data obtained from https://nijianmo.github.io/amazon/index.html.

Here nodes are Amazon reviewers and hyperedges correspond to reviewers who
reviewed a particular type of product within a month.  The product types are
different types of blues music.

The file hyperedges.txt contains lists of reviewers that reviewed a certain type
of product within a month. Each line lists the reviewer numbers that appeared
together in a hyperedge.

The file hyperedge-labels.txt lists the category type label (product type)
corresponding to each line in the hyperedges.txt file.

The file hyperedge-label-identities.txt lists the names of product categories,
with order in the list corresponding to the number label used in the file
hyperedge-labels.txt.

The file temporal-list.txt has format "(reviewer id) (category id) (timestamp)\n".
