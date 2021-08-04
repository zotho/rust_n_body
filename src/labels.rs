#[macro_export]
macro_rules! labels {
    ($ui:ident, $($fmt:expr, $($val:expr),+);+ $(;)?) => (
        $(
            $ui.label(None, format!($fmt, $($val,)+).as_str());
        )+
    )
}
