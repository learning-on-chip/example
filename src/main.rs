extern crate arguments;
extern crate configuration;
extern crate term;

#[macro_use] extern crate log;
#[macro_use] extern crate streamer;

use configuration::format::TOML;
use log::LogLevel;
use streamer::{Config, Result};
use streamer::{platform, schedule, system, traffic, workload};

mod logger;

use logger::Logger;

type System = system::System<traffic::Fractal,
                             workload::Random,
                             platform::Thermal,
                             schedule::Impartial>;

const USAGE: &'static str = "
Usage: example [options]

Options:
    --config <path>          Configuration file (required).

    --verbose                Display progress information.
    --help                   Display this message.
";

#[allow(unused_must_use)]
fn main() {
    start().unwrap_or_else(|error| {
        use std::io::Write;
        if let Some(mut output) = term::stderr() {
            output.fg(term::color::RED);
            output.write_fmt(format_args!("Error: {}.\n", error));
            output.reset();
        }
        std::process::exit(1);
    });
}

fn start() -> Result<()> {
    let arguments = ok!(arguments::parse(std::env::args()));
    if arguments.get::<bool>("help").unwrap_or(false) {
        println!("{}", USAGE.trim());
        return Ok(());
    }
    if arguments.get::<bool>("verbose").unwrap_or(false) {
        Logger::install(LogLevel::Info);
    } else {
        Logger::install(LogLevel::Warn);
    }

    let config = ok!(TOML::open(some!(arguments.get::<String>("config"),
                                      "a configuration file is required")));

    let mut system = try!(setup(&config));

    info!(target: "Example", "Simulating...");
    while let Some((_, _)) = try!(system.next()) {
        break;
    }
    info!(target: "Example", "Well done.");

    Ok(())
}

fn setup(config: &Config) -> Result<System> {
    macro_rules! branch(($name:expr) => (config.branch($name).as_ref().unwrap_or(config)));

    let source = streamer::source(config);
    let traffic = try!(traffic::Fractal::new(branch!("traffic"), &source));
    let workload = try!(workload::Random::new(branch!("workload"), &source));
    let platform = try!(platform::Thermal::new(branch!("platform")));
    let schedule = try!(schedule::Impartial::new(branch!("schedule"), &platform));

    System::new(traffic, workload, platform, schedule)
}
