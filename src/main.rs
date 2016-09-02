extern crate arguments;
extern crate configuration;
extern crate sql;
extern crate sqlite;
extern crate term;

#[macro_use] extern crate log;
#[macro_use] extern crate streamer;

use configuration::format::TOML;
use log::LogLevel;
use streamer::{Result, platform, schedule, traffic, workload};
use streamer::system::{self, Event};

mod logger;
mod output;

use logger::Logger;
use output::Output;

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
    if let Err(error) = start() {
        use std::io::Write;
        if let Some(mut output) = term::stderr() {
            output.fg(term::color::RED);
            output.write_fmt(format_args!("Error: {}.\n", error));
            output.reset();
        }
        std::process::exit(1);
    }
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
    macro_rules! branch(($name:expr) => (config.branch($name).as_ref().unwrap_or(&config)));
    let mut system = {
        let source = streamer::source(&config);
        let traffic = try!(traffic::Fractal::new(branch!("traffic"), source.clone()));
        let workload = try!(workload::Random::new(branch!("workload"), source.clone()));
        let platform = try!(platform::Thermal::new(branch!("platform")));
        let schedule = try!(schedule::Impartial::new(branch!("schedule"), &platform, source));
        try!(System::new(traffic, workload, platform, schedule))
    };
    let time_span = *some!(config.get::<f64>("output.time_span"), "a time span is required");
    let mut output = try!(Output::new(system.platform(), branch!("output")));
    info!(target: "Example", "Synthesizing {} seconds...", time_span);
    while let Some((event, data)) = try!(system.next()) {
        if event.time > time_span {
            break;
        }
        display(&system, &event);
        try!(output.next(&event, &data));
    }
    info!(target: "Example", "Well done.");
    Ok(())
}

fn display(system: &System, event: &Event) {
    use streamer::system::EventKind;

    let (job, kind) = match &event.kind {
        &EventKind::Arrive(ref job) => (job, "arrive"),
        &EventKind::Start(ref job, _) => (job, "start"),
        &EventKind::Finish(ref job, _) => (job, "finish"),
    };
    info!(target: "Example",
          "{:10.2} s | {:6} | # {:<5} ( {:15} | {:2} components | {:6.2} s ) {:2} queued",
          event.time, kind, job.id, shorten(&job.name, 15), job.component_count, job.duration(),
          system.history().arrived - system.history().started);
}

fn shorten(line: &str, limit: usize) -> String {
    if line.len() <= limit {
        return line.to_string();
    }
    let mut line = line[0..(limit - 1)].to_string();
    line.push('…');
    line
}
