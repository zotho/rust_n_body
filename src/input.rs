#[macro_export]
macro_rules! join {
    ($first:tt) => {
        $first
    };

    ($first:tt, $( $rest:tt ),+) => {
        $first | join!($rest)
    };
}

#[macro_export]
macro_rules! handle_input {
    ($method:ident, $key:expr, $action:expr) => (
        if $method($key) {
            $action
        }
    );

    ($method:ident; [$key:expr => $action:expr]) => (
        handle_input!($method, $key, $action);
    );

    ($method:ident; [$key:expr => $action:expr; $($rest:tt)*]) => (
        handle_input!($method, $key, $action);
        handle_input!($method; [$($rest)*]);
    );

    ($( $method:ident => { $( $a:expr => $b:expr $(;)?)+ } )+ ) => (
        $(
            handle_input!($method; [$($a => $b);+]);
        )+
    );

    // ($method:ident => { $( $key:expr => $action:expr );+ $(;)? } ) => (
    //     $(
    //         handle_input!($method, $key, $action);
    //     )+
    // );

    // (#[$proc_macro:expr] $method:ident => { $( $key:expr => $action:expr );+ $(;)? } ) => {
    //     $(
    //         #[$proc_macro]
    //         handle_input!($method, $key, $action);
    //     )+
    // }
}

/*
handle_input!(
    KeyCode::Left => time_speed /= 1.1;
    MouseButton::Left => {
        let (px, py) = mouse_pos_prev;
        let (px, py) = (px as f64, py as f64);
        ...
    }
)
*/
