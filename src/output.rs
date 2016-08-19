use sqlite::{Connection, Statement, State};
use std::mem;

use streamer::{Config, Result};
use streamer::platform::{self, Platform, Profile};
use streamer::system::Event;

pub struct Output {
    #[allow(dead_code)]
    connection: Connection,
    statement: Statement<'static>,
    buffer: (usize, usize, Vec<f64>, Vec<f64>),
}

impl Output {
    pub fn new(platform: &platform::Thermal, config: &Config) -> Result<Self> {
        use sql::prelude::*;

        let connection = ok!(Connection::open(path!(@unchecked config,
                                                    "an output file is required")));
        ok!(connection.execute("
            PRAGMA journal_mode = MEMORY;
            PRAGMA synchronous = OFF;
        "));
        ok!(connection.execute(
            ok!(create_table("profiles").if_not_exists().columns(&[
                "time".float().not_null(), "component_id".integer().not_null(),
                "power".float().not_null(), "temperature".float().not_null(),
            ]).compile())
        ));
        ok!(connection.execute(ok!(delete_from("profiles").compile())));
        let units = platform.elements().len();
        let statement = {
            let statement = ok!(connection.prepare(
                ok!(insert_into("profiles").columns(&[
                    "time", "component_id", "power", "temperature",
                ]).batch(units).compile())
            ));
            unsafe { mem::transmute(statement) }
        };
        let subsample = *config.get::<i64>("subsample").unwrap_or(&1);
        if subsample <= 0 {
            raise!("the subsample size should be greater than zero");
        }
        Ok(Output {
            connection: connection,
            statement: statement,
            buffer: (subsample as usize, 0, vec![0.0; units], vec![0.0; units]),
        })
    }

    pub fn next(&mut self, _: &Event, profiles: &(Profile, Profile)) -> Result<()> {
        ok!(self.connection.execute("BEGIN TRANSACTION"));
        ok!(self.write_profiles(profiles));
        ok!(self.connection.execute("END TRANSACTION"));
        Ok(())
    }

    fn write_profiles(&mut self, profiles: &(Profile, Profile)) -> Result<()> {
        let &Profile { units, steps, time, time_step, data: ref new_power } = &profiles.0;
        let &Profile { data: ref new_temperature, .. } = &profiles.1;
        let &mut Output { ref mut statement, ref mut buffer, .. } = self;
        let &mut (subsample, ref mut position, ref mut power, ref mut temperature) = buffer;
        for i in 0..steps {
            for j in 0..units {
                power[j] += new_power[i * units + j];
                temperature[j] += new_temperature[i * units + j];
            }
            *position = (*position + 1) % subsample;
            if *position > 0 {
                continue;
            }
            let time = time + (i as f64) * time_step;
            ok!(statement.reset());
            let mut k = 0;
            for j in 0..units {
                power[j] /= subsample as f64;
                temperature[j] /= subsample as f64;
                ok!(statement.bind(k + 1, time));
                ok!(statement.bind(k + 2, j as i64));
                ok!(statement.bind(k + 3, power[j]));
                ok!(statement.bind(k + 4, temperature[j]));
                power[j] = 0.0;
                temperature[j] = 0.0;
                k += 4;
            }
            if State::Done != ok!(statement.next()) {
                raise!("failed to write into the database");
            }
        }
        Ok(())
    }
}
