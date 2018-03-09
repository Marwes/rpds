/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

use std::vec::Vec;
use std::sync::Arc;
use std::borrow::Borrow;
use std::fmt::Display;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::iter::FromIterator;
use std::mem::size_of;

use std::rc::Rc;
use std::ops::Deref;

pub trait RefPtr: Clone + Deref {
    fn new(value: Self::Target) -> Self;
    fn make_mut(self_: &mut Self) -> &mut Self::Target
    where
        Self::Target: Clone;
}

impl<T> RefPtr for Rc<T> {
    fn new(value: Self::Target) -> Self {
        Rc::new(value)
    }
    fn make_mut(self_: &mut Self) -> &mut Self::Target
    where
        T: Clone,
    {
        Rc::make_mut(self_)
    }
}

impl<T> RefPtr for Arc<T> {
    fn new(value: Self::Target) -> Self {
        Arc::new(value)
    }
    fn make_mut(self_: &mut Self) -> &mut Self::Target
    where
        T: Clone,
    {
        Arc::make_mut(self_)
    }
}

pub trait Shared<T> {
    type Ptr: RefPtr<Target = T>;

    fn new(value: <Self::Ptr as Deref>::Target) -> Self::Ptr {
        Self::Ptr::new(value)
    }

    fn make_mut(self_: &mut Self::Ptr) -> &mut <Self::Ptr as Deref>::Target
    where
        T: Clone,
    {
        Self::Ptr::make_mut(self_)
    }
}

impl<T> Shared<T> for Rc<()> {
    type Ptr = Rc<T>;
}

impl<T> Shared<T> for Arc<()> {
    type Ptr = Arc<T>;
}

// TODO Use impl trait instead of this when available.
pub type Iter<'a, T, P> = ::std::iter::Map<IterArc<'a, T, P>, fn(&<P as Shared<T>>::Ptr) -> &T>;

const DEFAULT_BITS: u8 = 5;

/// Creates a [`Vector`](sequence/vector/struct.Vector.html) containing the given arguments:
///
/// ```
/// # use rpds::*;
/// #
/// let v = Vector::new()
///     .push_back(1)
///     .push_back(2)
///     .push_back(3);
///
/// assert_eq!(vector![1, 2, 3], v);
/// ```
#[macro_export]
macro_rules! vector {
    ($($e:expr),*) => {
        {
            #[allow(unused_mut)]
            let mut v = $crate::Vector::new();
            $(
                v.push_back_mut($e);
            )*
            v
        }
    };
}

/// A persistent vector with structural sharing.  This data structure supports fast `push_back()`, `set()`,
/// `drop_last()`, and `get()`.
///
/// # Complexity
///
/// Let *n* be the number of elements in the vector.
///
/// ## Temporal complexity
///
/// | Operation                  | Best case | Average   | Worst case  |
/// |:-------------------------- | ---------:| ---------:| -----------:|
/// | `new()`                    |      Θ(1) |      Θ(1) |        Θ(1) |
/// | `set()`                    | Θ(log(n)) | Θ(log(n)) |   Θ(log(n)) |
/// | `push_back()`              | Θ(log(n)) | Θ(log(n)) |   Θ(log(n)) |
/// | `drop_last()`              | Θ(log(n)) | Θ(log(n)) |   Θ(log(n)) |
/// | `first()`/`last()`/`get()` | Θ(log(n)) | Θ(log(n)) |   Θ(log(n)) |
/// | `len()`                    |      Θ(1) |      Θ(1) |        Θ(1) |
/// | `clone()`                  |      Θ(1) |      Θ(1) |        Θ(1) |
/// | iterator creation          |      Θ(1) |      Θ(1) |        Θ(1) |
/// | iterator step              |      Θ(1) |      Θ(1) |   Θ(log(n)) |
/// | iterator full              |      Θ(n) |      Θ(n) |        Θ(n) |
///
/// ### Proof sketch of the complexity of full iteration
///
/// 1. A tree of size *n* and degree *d* has height *⌈log<sub>d</sub>(n)⌉ - 1*.
/// 2. A complete iteration is a depth-first search on the tree.
/// 3. A depth-first search has complexity *Θ(|V| + |E|)*, where *|V|* is the number of nodes and
///    *|E|* the number of edges.
/// 4. The number of nodes *|V|* for a complete tree of height *h* is the sum of powers of *d*, which is
///    *(dʰ - 1) / (d - 1)*. See
///    [Calculating sum of consecutive powers of a number](https://math.stackexchange.com/questions/971761/calculating-sum-of-consecutive-powers-of-a-number).
/// 5. The number of edges is exactly *|V| - 1*.
///
/// By 2. and 3. we have that the complexity of a full iteration is
///
/// ```text
///      Θ(|V| + |E|)
///    = Θ((dʰ - 1) / (d - 1))      (by 4. and 5.)
///    = Θ(dʰ)
///    = Θ(n)                       (by 1.)
/// ```
///
/// # Implementation details
///
/// This vector is implemented as described in
/// [Understanding Persistent Vector Part 1](http://hypirion.com/musings/understanding-persistent-vector-pt-1)
/// and [Understanding Persistent Vector Part 2](http://hypirion.com/musings/understanding-persistent-vector-pt-2).
pub struct SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    root: <P as Shared<Node<T, P>>>::Ptr,
    bits: u8,
    length: usize,
}

impl<T, P> ::std::fmt::Debug for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
    T: ::std::fmt::Debug,
{
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        unimplemented!()
    }
}

pub type Vector<T> = SharedVector<T, Arc<()>>;

#[doc(hidden)]
pub enum Node<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    Branch(Vec<<P as Shared<Node<T, P>>>::Ptr>),
    Leaf(Vec<<P as Shared<T>>::Ptr>),
}

impl<T, P> ::std::fmt::Debug for Node<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
    T: Display,
{
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        unimplemented!()
    }
}

impl<T: Eq, P> Eq for Node<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
}

impl<T: PartialEq, P> PartialEq for Node<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn eq(&self, other: &Node<T, P>) -> bool {
        unimplemented!()
    }
}

impl<T, P> Node<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn new_empty_branch() -> Node<T, P> {
        Node::Branch(Vec::new())
    }

    fn new_empty_leaf() -> Node<T, P> {
        Node::Leaf(Vec::new())
    }

    fn get<F: Fn(usize, usize) -> usize>(&self, index: usize, height: usize, bucket: F) -> &T {
        let b = bucket(index, height);

        match *self {
            Node::Branch(ref a) => a[b].get(index, height - 1, bucket),
            Node::Leaf(ref a) => {
                debug_assert_eq!(height, 0);
                &a[b]
            }
        }
    }

    fn assoc<F: Fn(usize) -> usize>(&mut self, value: T, height: usize, bucket: F) {
        let b = bucket(height);

        match *self {
            Node::Leaf(ref mut a) => {
                debug_assert_eq!(height, 0, "cannot have a leaf at this height");

                let value = <P as Shared<T>>::new(value);
                if a.len() == b {
                    a.push(value);
                } else {
                    a[b] = value;
                }
            }

            Node::Branch(ref mut a) => {
                debug_assert!(height > 0, "cannot have a branch at this height");

                if let Some(subtree) = a.get_mut(b) {
                    <P as Shared<Node<_, _>>>::make_mut(subtree).assoc(value, height - 1, bucket);
                    return;
                }
                let mut subtree = if height > 1 {
                    Node::new_empty_branch()
                } else {
                    Node::new_empty_leaf()
                };

                subtree.assoc(value, height - 1, bucket);
                a.push(<P as Shared<Node<_, _>>>::new(subtree));
            }
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.used() == 0
    }

    #[inline]
    fn is_singleton(&self) -> bool {
        self.used() == 1
    }

    fn used(&self) -> usize {
        match *self {
            Node::Leaf(ref a) => a.len(),
            Node::Branch(ref a) => a.len(),
        }
    }

    /// Drops the last element.
    ///
    /// This will return `None` if the subtree after drop becomes empty (or it already was empty).
    /// Note that this will prune irrelevant branches, i.e. there will be no branches without
    /// elements under it.
    fn drop_last(&mut self) -> Option<()> {
        if self.is_empty() {
            return None;
        }

        match *self {
            Node::Leaf(ref mut a) => {
                a.pop();
            }

            Node::Branch(ref mut a) => {
                match <P as Shared<Node<_, _>>>::Ptr::make_mut(a.last_mut().unwrap()).drop_last() {
                    Some(()) => (),
                    None => {
                        a.pop();
                    }
                }
            }
        }

        if self.is_empty() {
            None
        } else {
            Some(())
        }
    }
}

impl<T, P> Clone for Node<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn clone(&self) -> Node<T, P> {
        match *self {
            Node::Branch(ref a) => Node::Branch(Vec::clone(a)),
            Node::Leaf(ref a) => Node::Leaf(Vec::clone(a)),
        }
    }
}

impl<T, P> SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    pub fn new() -> SharedVector<T, P> {
        SharedVector::new_with_bits(DEFAULT_BITS)
    }

    pub fn new_with_bits(bits: u8) -> SharedVector<T, P> {
        assert!(bits > 0, "number of bits for the vector must be positive");

        SharedVector {
            root: <P as Shared<Node<_, _>>>::new(Node::new_empty_leaf()),
            bits,
            length: 0,
        }
    }

    #[inline]
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    pub fn last(&self) -> Option<&T> {
        match self.length {
            0 => None,
            n => self.get(n - 1),
        }
    }

    #[inline]
    fn degree(&self) -> usize {
        1 << self.bits
    }

    #[inline]
    fn height(&self) -> usize {
        if self.length > 1 {
            let u: usize = self.length - 1;
            let c: usize = 8 * size_of::<usize>() - u.leading_zeros() as usize;
            let b: usize = self.bits as usize;

            c / b + if c % b > 0 { 1 } else { 0 } - 1
        } else {
            0
        }
    }

    #[inline]
    fn mask(&self) -> usize {
        self.degree() - 1
    }

    #[inline]
    fn bucket(&self, index: usize, height: usize) -> usize {
        (index >> (height * self.bits as usize)) & self.mask()
    }

    #[inline]
    fn bucket2(bits: u8, index: usize, height: usize) -> usize {
        (index >> (height * bits as usize)) & ((1 << bits) - 1)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.length {
            None
        } else {
            Some(self.root.get(index, self.height(), |index, height| {
                self.bucket(index, height)
            }))
        }
    }

    pub fn set(&self, index: usize, v: T) -> Option<SharedVector<T, P>> {
        let mut self_ = self.clone();
        self_.set_mut(index, v).map(|()| self_)
    }

    pub fn set_mut(&mut self, index: usize, v: T) -> Option<()> {
        if index >= self.length {
            None
        } else {
            self.assoc(index, v);
            Some(())
        }
    }

    /// Sets the given index to v.
    ///
    /// # Panics
    ///
    /// This method will panic if the trie's root doesn't have capacity for the given index.
    fn assoc(&mut self, index: usize, v: T) {
        debug_assert!(
            index < self.root_max_capacity(),
            "This trie's root cannot support this index"
        );

        let height = self.height();
        let bits = self.bits;
        <P as Shared<Node<_, _>>>::make_mut(&mut self.root)
            .assoc(v, height, |height| Self::bucket2(bits, index, height));
        let adds_item: bool = index >= self.length;

        self.bits = bits;
        self.length += if adds_item { 1 } else { 0 };
    }

    #[inline]
    fn root_max_capacity(&self) -> usize {
        /* A Trie's root max capacity is
         *
         *     degree ** (height + 1)
         *   = (2 ** bits) ** (height + 1)        (by def. of degree)
         *   = 2 ** (bits * (height + 1))
         *   = 1 << (bits * (height + 1))
         */
        1 << (self.bits as usize * (self.height() + 1))
    }

    #[inline]
    fn is_root_full(&self) -> bool {
        self.length == self.root_max_capacity()
    }

    pub fn push_back(&self, v: T) -> SharedVector<T, P> {
        let mut self_ = self.clone();
        self_.push_back_mut(v);
        self_
    }

    pub fn push_back_mut(&mut self, v: T) {
        if self.is_root_full() {
            let mut new_root: Node<T, P> = Node::new_empty_branch();

            match new_root {
                Node::Branch(ref mut values) => values.push(self.root.clone()),
                _ => unreachable!("expected a branch"),
            }

            let length = self.length;
            self.root = <P as Shared<Node<_, _>>>::new(new_root);
            self.length += 1;

            self.assoc(length, v)
        } else {
            let length = self.length;
            self.assoc(length, v)
        }
    }

    /// Compresses a root.  A root is compressed if, whenever there is a branch, it has more than
    /// one child.
    ///
    /// The trie must always have a compressed root.
    #[cfg(test)]
    fn compress_root(mut root: Node<T, P>) -> <P as Shared<Node<T, P>>>::Ptr {
        match Self::compress_root_mut(&mut root) {
            Some(new_root) => new_root,
            None => <P as Shared<Node<_, _>>>::new(root),
        }
    }

    fn compress_root_mut(root: &mut Node<T, P>) -> Option<<P as Shared<Node<T, P>>>::Ptr> {
        match *root {
            Node::Leaf(_) => None,
            Node::Branch(_) => if root.is_singleton() {
                if let Node::Branch(ref mut a) = *root {
                    a.pop()
                } else {
                    unreachable!()
                }
            } else {
                None
            },
        }
    }

    pub fn drop_last(&self) -> Option<SharedVector<T, P>> {
        let mut self_ = self.clone();
        self_.drop_last_mut().map(|()| self_)
    }

    pub fn drop_last_mut(&mut self) -> Option<()> {
        if self.length == 0 {
            return None;
        }

        let new_root = {
            let root = <P as Shared<Node<_, _>>>::make_mut(&mut self.root);
            root.drop_last();
            self.length -= 1;
            SharedVector::compress_root_mut(root)
        };

        if let Some(new_root) = new_root {
            self.root = new_root;
        }

        Some(())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> Iter<T, P> {
        self.iter_arc().map(|v| v.borrow())
    }

    fn iter_arc(&self) -> IterArc<T, P> {
        IterArc::new(self)
    }
}

impl<T, P> Index<usize> for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        self.get(index)
            .expect(&format!("index out of bounds {}", index))
    }
}

impl<T, P> Default for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn default() -> SharedVector<T, P> {
        Self::new()
    }
}

impl<T: PartialEq, P> PartialEq for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn eq(&self, other: &SharedVector<T, P>) -> bool {
        self.length == other.length && self.iter().eq(other.iter())
    }
}

impl<T: Eq, P> Eq for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
}

impl<T: PartialOrd, P> PartialOrd for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn partial_cmp(&self, other: &SharedVector<T, P>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: Ord, P> Ord for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn cmp(&self, other: &SharedVector<T, P>) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<T: Hash, P> Hash for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn hash<H: Hasher>(&self, state: &mut H) -> () {
        // Add the hash of length so that if two collections are added one after the other it doesn't
        // hash to the same thing as a single collection with the same elements in the same order.
        self.len().hash(state);

        for e in self {
            e.hash(state);
        }
    }
}

impl<T, P> Clone for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn clone(&self) -> SharedVector<T, P> {
        SharedVector {
            root: self.root.clone(),
            bits: self.bits,
            length: self.length,
        }
    }
}

impl<T, P> Display for SharedVector<T, P>
where
    T: Display,
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        let mut first = true;

        fmt.write_str("[")?;

        for v in self.iter() {
            if !first {
                fmt.write_str(", ")?;
            }
            v.fmt(fmt)?;
            first = false;
        }

        fmt.write_str("]")
    }
}

impl<'a, T, P> IntoIterator for &'a SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T, P>;

    fn into_iter(self) -> Iter<'a, T, P> {
        self.iter()
    }
}

impl<T, P> FromIterator<T> for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn from_iter<I: IntoIterator<Item = T>>(into_iter: I) -> SharedVector<T, P> {
        let mut vector = SharedVector::new();
        vector.extend(into_iter);
        vector
    }
}

impl<T, P> Extend<T> for SharedVector<T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for elem in iter {
            self.push_back_mut(elem);
        }
    }
}

pub struct IterArc<'a, T: 'a, P>
where
    P: Shared<Node<T, P>> + Shared<T> + 'a,
{
    vector: &'a SharedVector<T, P>,

    stack_forward: Option<Vec<IterStackElement<'a, T, P>>>,
    stack_backward: Option<Vec<IterStackElement<'a, T, P>>>,

    left_index: usize,  // inclusive
    right_index: usize, // exclusive
}

struct IterStackElement<'a, T: 'a, P>
where
    P: Shared<Node<T, P>> + Shared<T> + 'a,
{
    node: &'a Node<T, P>,
    index: isize,
}

impl<'a, T, P> IterStackElement<'a, T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn new(node: &Node<T, P>, backwards: bool) -> IterStackElement<T, P> {
        IterStackElement {
            node,
            index: if backwards {
                node.used() as isize - 1
            } else {
                0
            },
        }
    }

    fn current_node(&self) -> &'a Node<T, P> {
        match *self.node {
            Node::Branch(ref a) => &a[self.index as usize],
            Node::Leaf(_) => panic!("called current node of a branch"),
        }
    }

    fn current_elem(&self) -> &'a <P as Shared<T>>::Ptr {
        match *self.node {
            Node::Leaf(ref a) => &a[self.index as usize],
            Node::Branch(_) => panic!("called current element of a branch"),
        }
    }

    /// Advance and returns `true` if finished.
    #[inline]
    fn advance(&mut self, backwards: bool) -> bool {
        if backwards {
            self.index -= 1;
            self.index < 0
        } else {
            self.index += 1;
            self.index as usize >= self.node.used()
        }
    }
}

impl<'a, T, P> IterArc<'a, T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn new(vector: &SharedVector<T, P>) -> IterArc<T, P> {
        IterArc {
            vector,

            stack_forward: None,
            stack_backward: None,

            left_index: 0,
            right_index: vector.len(),
        }
    }

    fn dig(stack: &mut Vec<IterStackElement<T, P>>, backwards: bool) -> () {
        let next_node: &Node<T, P> = {
            let stack_top = stack.last().unwrap();

            if let Node::Leaf(_) = *stack_top.node {
                return;
            }

            stack_top.current_node()
        };

        stack.push(IterStackElement::new(next_node, backwards));

        IterArc::dig(stack, backwards);
    }

    fn init_if_needed(&mut self, backwards: bool) -> () {
        let stack_field = if backwards {
            &mut self.stack_backward
        } else {
            &mut self.stack_forward
        };

        if stack_field.is_none() {
            let mut stack: Vec<IterStackElement<T, P>> =
                Vec::with_capacity(self.vector.height() + 1);

            stack.push(IterStackElement::new(self.vector.root.borrow(), backwards));

            IterArc::dig(&mut stack, backwards);

            *stack_field = Some(stack);
        }
    }

    fn advance(stack: &mut Vec<IterStackElement<T, P>>, backwards: bool) -> () {
        match stack.pop() {
            Some(mut stack_element) => {
                let finished = stack_element.advance(backwards);

                if finished {
                    IterArc::advance(stack, backwards);
                } else {
                    stack.push(stack_element);

                    IterArc::dig(stack, backwards);
                }
            }
            None => (), // Reached the end.  Nothing to do.
        }
    }

    #[inline]
    fn current(stack: &[IterStackElement<'a, T, P>]) -> Option<&'a <P as Shared<T>>::Ptr> {
        stack.last().map(|e| e.current_elem())
    }

    #[inline]
    fn non_empty(&self) -> bool {
        self.left_index < self.right_index
    }

    fn advance_forward(&mut self) -> () {
        if self.non_empty() {
            IterArc::advance(self.stack_forward.as_mut().unwrap(), false);

            self.left_index += 1;
        }
    }

    fn current_forward(&self) -> Option<&'a <P as Shared<T>>::Ptr> {
        if self.non_empty() {
            IterArc::current(self.stack_forward.as_ref().unwrap())
        } else {
            None
        }
    }

    fn advance_backward(&mut self) -> () {
        if self.non_empty() {
            IterArc::advance(self.stack_backward.as_mut().unwrap(), true);

            self.right_index -= 1;
        }
    }

    fn current_backward(&self) -> Option<&'a <P as Shared<T>>::Ptr> {
        if self.non_empty() {
            IterArc::current(self.stack_backward.as_ref().unwrap())
        } else {
            None
        }
    }
}

impl<'a, T, P> Iterator for IterArc<'a, T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    type Item = &'a <P as Shared<T>>::Ptr;

    fn next(&mut self) -> Option<&'a <P as Shared<T>>::Ptr> {
        self.init_if_needed(false);

        let current = self.current_forward();

        self.advance_forward();

        current
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.right_index - self.left_index;

        (len, Some(len))
    }
}

impl<'a, T, P> DoubleEndedIterator for IterArc<'a, T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
    fn next_back(&mut self) -> Option<&'a <P as Shared<T>>::Ptr> {
        self.init_if_needed(true);

        let current = self.current_backward();

        self.advance_backward();

        current
    }
}

impl<'a, T, P> ExactSizeIterator for IterArc<'a, T, P>
where
    P: Shared<Node<T, P>> + Shared<T>,
{
}

#[cfg(feature = "serde")]
pub mod serde {
    use super::*;
    use serde::ser::{Serialize, Serializer};
    use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
    use std::marker::PhantomData;
    use std::fmt;

    impl<T, P> Serialize for SharedVector<T, P>
    where
        T: Serialize,
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self)
        }
    }

    impl<'de, T, P> Deserialize<'de> for SharedVector<T, P>
    where
        T: Deserialize<'de>,
    {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<SharedVector<T>, D::Error> {
            deserializer.deserialize_seq(VectorVisitor {
                phantom: PhantomData,
            })
        }
    }

    struct VectorVisitor<T> {
        phantom: PhantomData<T>,
    }

    impl<'de, T> Visitor<'de> for VectorVisitor<T>
    where
        T: Deserialize<'de>,
    {
        type Value = SharedVector<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<SharedVector<T>, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut vector = SharedVector::new();

            while let Some(value) = seq.next_element()? {
                vector.push_back_mut(value);
            }

            Ok(vector)
        }
    }
}

#[cfg(test)]
mod test;
