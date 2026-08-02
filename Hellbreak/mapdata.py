"""Level definitions for HELLBREAK.

Maps are ASCII grids.  Legend:

  walls:   '#' brick   'M' tech    '%' grey stone  '4' rust
           'B' bone    'L' hellrock  'C' crate
  doors:   'D' door    'U' blue-key door   'R' red-key door
  special: 'X' exit switch (activate to finish level)
  floor:   '.' empty
  spawn:   'P' player start (facing angle set per level)
  enemies: 'g' grunt   'i' fiend   'v' ravager
  items:   'h' stim    'H' medkit  'a' bullet clip  's' shell box
           'c' cell pack  'A' armor  'u' blue key    'r' red key
           '2' shotgun    '3' chaingun  '5' plasma rifle  'o' barrel
"""

WALL_CHARS = {"#": 1, "M": 2, "%": 3, "4": 4, "B": 5, "L": 6,
              "D": 7, "U": 8, "R": 9, "X": 10, "C": 12}
DOOR_CHARS = {"D": None, "U": "blue", "R": "red"}

LEVEL_1 = {
    "name": "TECHBASE PERIMETER",
    "floor": "concrete",
    "floor2": "hex",
    "ceiling": "metal",
    "player_angle": 0.0,
    "grid": [
        "MMMMMMMMMMMMMMMM%%%%%%%%%%%%",
        "M..............M%..........%",
        "M.P........a...D....g....h.%",
        "M..............M%..........%",
        "M...g......o...M%...C.C....%",
        "M..............M%..s.......%",
        "MMMMMM.MMMMMMMMM%%.%%%%%%%.%",
        "M....M.M.......M%..%####%..%",
        "M.h..M.M...g...M%..%#..#%..%",
        "M....D.M.......D...D...2#..%",
        "M....M.M...o...M%..%#..#%..%",
        "M.a..M.M.......M%..%####%..%",
        "M....M.MMMMMMMMM%%.%%%%%%%.%",
        "MMMMMM.M.......M%..........%",
        "M......M...g.g.M%.g......i.%",
        "M.MMMMMM.......M%..........%",
        "M.M....M..uH...M%%%%%%U%%%%%",
        "M.M.gg.M.......M#....o.....#",
        "M.M....MMMMMDMMM#..........#",
        "M.M.h..M...#...##.i......A.#",
        "M.MM.MMM.s.#...............#",
        "M........g.D...#h..........#",
        "M.MMMMMM...#...##....g.....#",
        "M.M....MM.MMMMMM#..........#",
        "M.Mgg..M4D44444M#####RRR####",
        "M.M....M4..3..4M#..........#",
        "M.D...MM4..r..4MM....X.....#",
        "M..M..MM4.....4MM..........#",
        "MMMMMMMM4444444M############",
    ],
}

LEVEL_2 = {
    "name": "GATES OF THE SPIRE",
    "floor": "blood",
    "floor2": "hex",
    "ceiling": "cave",
    "player_angle": 1.5708,
    "grid": [
        "LLLLLLLLBBBBBBBBBBLLLLLLLLLL",
        "L..........................L",
        "L.i...v..........v.......i.L",
        "L..........................L",
        "L...o..LLLL....LLLL....o...L",
        "L......L##D....D##L........L",
        "LL.L.LLL#..#..#..#LLLL.LLLLL",
        "B..L.L..#.s#..#a.#...L.L...B",
        "B..L.L..####..####...L.L...B",
        "B..LDL...............LDL...B",
        "B..L.L..g....g....g..L.L...B",
        "B..L.L...............L.L...B",
        "B..LLLLLLLLLULLLLLLLLL.L...B",
        "B.h....L....r....L.....L.A.B",
        "B......L.g.....i.L.....L...B",
        "BBBB.BBL.........LBBBBBBB.BB",
        "B..B.B.L....o....L.B......5B",
        "B.iB.B.LLLLLULLLLL.B.BBBB.BB",
        "B..B.B.....L.L.....D.B..B..B",
        "B..B.BBBBB.L.L.BBBB.c.B.B..B",
        "B..B.....B.LuL.B...B..B.i..B",
        "B..BBBBB.B.L.L.B.BBB.BB.BBBB",
        "B..o...B.B.LDL.B.B...BvvB..B",
        "BB.BBB.B.......B.B.BBB..D..B",
        "B..B...B.Bv...vB.B...BssB.HB",
        "B.PB.BBB.B.....B.BBB.BBBB.BB",
        "B..B.B...BB...BB...B.......B",
        "B..D.D.g..B.#.B..h.R...X...B",
        "B..B.B....B.#.B....B.......B",
        "BBBBBBBBBBBB#BBBBBBBBBBBBBBB",
    ],
}

LEVELS = [LEVEL_1, LEVEL_2]
