use sqlite::{Connection, Statement, State};
use std::mem;

use streamer::{Config, Result};
use streamer::platform::{self, Platform, Profile};
use streamer::system::{Event, EventKind, Job};

pub struct Output {
    #[allow(dead_code)]
    connection: Connection,
    arrivals: Statement<'static>,
    profiles: Statement<'static>,
    length: f64,
}

impl Output {
    pub fn new(platform: &platform::Thermal, config: &Config) -> Result<Self> {
        use sql::prelude::*;

        let length = match config.get::<f64>("length") {
            Some(length) => *length,
            _ => raise!("an output length is required"),
        };

        let connection = match config.get::<String>("path") {
            Some(path) => ok!(Connection::open(path)),
            _ => raise!("an output file is required"),
        };

        ok!(connection.execute("
            PRAGMA journal_mode = MEMORY;
            PRAGMA synchronous = OFF;
        "));

        ok!(connection.execute(
            ok!(create_table("arrivals").if_not_exists().columns(&[
                "time".float().not_null(),
            ]).compile())
        ));
        ok!(connection.execute(
            ok!(create_table("profiles").if_not_exists().columns(&[
                "time".float().not_null(), "component_id".integer().not_null(),
                "power".float().not_null(), "temperature".float().not_null(),
            ]).compile())
        ));

        ok!(connection.execute(ok!(delete_from("arrivals").compile())));
        ok!(connection.execute(ok!(delete_from("profiles").compile())));

        let arrivals = {
            let statement = ok!(connection.prepare(
                ok!(insert_into("arrivals").columns(&[
                    "time",
                ]).compile())
            ));
            unsafe { mem::transmute(statement) }
        };
        let units = platform.elements().len();
        let profiles = {
            let statement = ok!(connection.prepare(
                ok!(insert_into("profiles").columns(&[
                    "time", "component_id", "power", "temperature",
                ]).batch(units).compile())
            ));
            unsafe { mem::transmute(statement) }
        };

        Ok(Output {
            connection: connection,
            arrivals: arrivals,
            profiles: profiles,
            length: length,
        })
    }

    pub fn next(&mut self, event: &Event, &(ref power, ref temperature): &(Profile, Profile))
                -> Result<Option<()>> {

        if event.time > self.length {
            return Ok(None);
        }
        ok!(self.connection.execute("BEGIN TRANSACTION"));
        if let &EventKind::Arrived(ref job) = &event.kind {
            ok!(self.write_arrival(job));
        }
        ok!(self.write_profile(power, temperature));
        ok!(self.connection.execute("END TRANSACTION"));
        Ok(Some(()))
    }

    fn write_arrival(&mut self, job: &Job) -> Result<()> {
        let statement = &mut self.arrivals;
        ok!(statement.reset());
        ok!(statement.bind(1, job.arrival));
        if State::Done != ok!(statement.next()) {
            raise!("failed to write into the database");
        }
        Ok(())
    }

    fn write_profile(&mut self, power: &Profile, temperature: &Profile) -> Result<()> {
        let &Profile { units, steps, time, time_step, data: ref power } = power;
        let &Profile { data: ref temperature, .. } = temperature;
        let statement = &mut self.profiles;
        for i in 0..steps {
            let time = time + (i as f64) * time_step;
            ok!(statement.reset());
            let mut k = 0;
            for j in 0..units {
                ok!(statement.bind(k + 1, time));
                ok!(statement.bind(k + 2, j as i64));
                ok!(statement.bind(k + 3, power[i * units + j]));
                ok!(statement.bind(k + 4, temperature[i * units + j]));
                k += 4;
            }
            if State::Done != ok!(statement.next()) {
                raise!("failed to write into the database");
            }
        }
        Ok(())
    }
}
