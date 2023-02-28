import qualified Data.Bifunctor

newtype BackProp a f = BackProp (a, f)


main = do
    print "test"


prod :: BackProp a (a -> x) -> BackProp b (b -> y) -> BackProp (a, b) ((a, b) -> (x, y))
prod (BackProp (a, f)) (BackProp (b, g)) = BackProp ((a, b), Data.Bifunctor.bimap f g)


rep :: Num x => BackProp a (a -> x) -> BackProp (a, a) ((a, a) -> x)
rep (BackProp (a, f)) = BackProp ((a, a), \(n, m) -> f n + f m)

p1 :: BackProp (a, b) ((a, b) -> (c, d)) -> (a -> BackProp e (e -> a)) -> BackProp (e, b) ((e, b) -> (c, d))
p1 (BackProp ((a, b), f)) g = let (BackProp (c, g')) = g a in BackProp ((c, b), \(h, k) -> f (g' h, k))

p2 :: BackProp (a, b) ((a, b) -> (c, d)) -> (b -> BackProp e (e -> b)) -> BackProp (e, b) ((a, e) -> (c, d))
p2 (BackProp ((a, b), f)) g = let (BackProp (c, g')) = g b in BackProp ((c, b), \(h, k) -> f (h, g' k))


data Dfdx a b = Dfdx b (b -> a)
